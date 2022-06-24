""" External verifiers in existing works. """

import subprocess
import sys
import tempfile
from datetime import datetime
import random
import string
from pathlib import Path
from typing import Optional, Tuple, List

import torch
from torch import Tensor

sys.path.append(str(Path(__file__).resolve().parent.parent))

from art.acas import AcasNet


class ExtVerifier(object):
    """ Essentially a call wrapper to external verifier. The interaction is done through dumped files. """

    def __init__(self):
        self.fdir = Path(tempfile.gettempdir(), f'art-{self.__class__.__name__}')
        self.fdir.mkdir(parents=True, exist_ok=True)
        return

    def verify(self, lb: Tensor, ub: Tensor, net: AcasNet, task_name, *args, **kwargs) -> Tensor:
        """
        :param task_name: which safety property to reason about, the format is specific to individual verifier
        :return: Tensor containing counterexamples' inputs, if any
        """
        raise NotImplementedError()

    def _next_fname(self, ext: str = 'nnet'):
        """ Returns a random file path for dumping to disk. """
        while True:
            hash = ''.join(random.choices(string.ascii_uppercase + string.digits, k=5))
            prefix = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            fname = f'{prefix}-{hash}'
            if ext is not None:
                fname = f'{fname}.{ext}'

            fpath = Path(self.fdir, fname)
            if not fpath.is_file():
                # return only file name here
                return fname
        pass
    pass


class _CEX(object):
    """ Counterexample returned from Reluplex or ReluVal. """
    def __init__(self):
        self.inputs = []
        self.outputs = []

        self.inputs_normed = []
        self.outputs_normed = []

        self.nnet_outputs = []
        self.nnet_outputs_normed = []
        return

    def __str__(self):
        s = ['CEX:']
        if self.inputs_normed is None:
            # from ReluVal
            nums = [str(n) for n in self.inputs]
            nums_str = ' '.join(nums)
            s.append(f'Input: [{nums_str}]')
        else:
            # from Reluplex
            for idx, (i, n) in enumerate(zip(self.inputs, self.inputs_normed)):
                s.append('Input[%d] = %f. Normalized: %f' % (idx, i, n))
            s.append('')

        if self.outputs_normed is None:
            # from ReluVal
            nums = [str(n) for n in self.outputs]
            nums_str = ' '.join(nums)
            s.append(f'Output: [{nums_str}]')
        else:
            # from Reluplex
            for idx, (o, n) in enumerate(zip(self.outputs, self.outputs_normed)):
                s.append('Output[%d] = %f. Normalized: %f' % (idx, o, n))
            s.append('')

            s.append('Output using nnet:')
            for idx, (o, n) in enumerate(zip(self.nnet_outputs, self.nnet_outputs_normed)):
                s.append('Output[%d] = %f. Normalized: %f' % (idx, o, n))
            s.append('')

        return '\n'.join(s)

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        if not isinstance(other, _CEX):
            return False
        return self.inputs == other.inputs
    pass


class Reluplex(ExtVerifier):
    """ Essentially a call wrapper to Reluplex (CAV'17). """

    def __init__(self, objdir: str = '/Users/xuankang/Workspace/ReluplexCav2017/bin'):
        super().__init__()
        self.objdir = objdir
        return

    def verify(self, lb: Tensor, ub: Tensor, net: AcasNet, task_name: str, use_normed: bool = True,
               *args, **kwargs) -> Tensor:
        """
        :param task_name: binary file for specific safety property
        """
        fname = self._next_fname()
        assert lb.size()[0] == 1 and ub.size()[0] == 1, 'Reluplex does not support multiple LB/UB bounds at once.'
        net.save(Path(self.fdir, fname), lb[0], ub[0])

        prop_path = Path(self.objdir, task_name)
        assert prop_path.is_file(), f'Bin not found for {task_name} at {prop_path}.'

        cmds = [prop_path]
        cmds.extend(args)

        # I was using Popen() and then wait(), but it somehow freezes there for local robustness property?!
        # While the following check_output() works.
        output = subprocess.check_output(cmds, stderr=subprocess.STDOUT)
        outs = output.decode()  # somehow the returned string is binary, needs decode()
        cexs = self.extract(outs)

        # reconstruct tensors based on cexs
        if use_normed:
            data = [c.inputs_normed for c in cexs]
        else:
            data = [c.inputs for c in cexs]
        return torch.tensor(data)

    def extract(self, logs: str) -> List[_CEX]:
        """ Extract Counterexamples returned in logs. """
        lines = logs.splitlines()
        cexs = []
        while True:
            c, lines = self._from_lines(lines)
            if c is None:
                break

            cexs.append(c)
        return cexs

    def _from_lines(self, lines: List[str]) -> Tuple[Optional[_CEX], List[str]]:
        """ Collect the first counterexample from the lines.
        :return: <CEX, remaining lines that may have more CEXs>
        """
        assert 'Aborted (core dumped)' not in lines, 'The lines are not terminating correctly?! Saw Aborted.'

        curr = None
        for i in range(len(lines)):
            if lines[i] == 'Solution found!':
                curr = i
                break

        if curr is None:
            # no such "Solution found!" text found
            return None, []  # [] because no more lines could have a CEX

        c = _CEX()
        curr += 1  # one empty line

        c.inputs = []
        c.inputs_normed = []
        curr += 1
        while len(lines[curr]) > 0:
            orig, norm = self._parse_cex_number_line(lines[curr])
            c.inputs.append(orig)
            c.inputs_normed.append(norm)
            curr += 1

        c.outputs = []
        c.outputs_normed = []
        curr += 1
        while len(lines[curr]) > 0:
            orig, norm = self._parse_cex_number_line(lines[curr])
            c.outputs.append(orig)
            c.outputs_normed.append(norm)
            curr += 1

        curr += 1
        assert lines[curr] == 'Output using nnet:' or lines[curr] == 'Output using nnet.cpp:'
        c.nnet_outputs = []
        c.nnet_outputs_normed = []

        curr += 1
        while len(lines[curr]) > 0:
            orig, norm = self._parse_cex_number_line(lines[curr])
            c.nnet_outputs.append(orig)
            c.nnet_outputs_normed.append(norm)
            curr += 1
        return c, lines[curr:]

    @staticmethod
    def _parse_cex_number_line(line):
        """ For number line in the shape of 'input[0] = 55947.690997. Normalized: 0.600000.'
        :return: <55947.690997, 0.6>
        """
        parts = line.split('. Normalized: ', maxsplit=1)

        orig = parts[0].split('=')[-1].strip()
        orig = float(orig)

        norm = parts[1].strip()
        if norm.endswith('.'):  # true for input, false for output (=.=)
            norm = norm[:-1]
        norm = float(norm)
        return orig, norm
    pass


class ReluVal(ExtVerifier):
    """ Essentially a call wrapper to ReluVal (USENIX'18).
        Successfully configured on ubuntu, so may have to query by ssh (assume ssh key placed there).
        Note that ReluVal's forward computation could have non-ignorable differences from my PyTorch code (~10^-2).
    """

    def __init__(self, objpath='/home/xuankang/Dropbox/workspace/ReluVal/quantify',
                 ssh_ip='68.60.241.24', ssh_user='xuankang'):
        super().__init__()

        self.objpath = Path(objpath)
        self.ssh_ip = ssh_ip
        self.ssh_user = ssh_user
        return

    def verify(self, lb: Tensor, ub: Tensor, net: AcasNet, task_name, collect_regions: bool = False,
               hide_err: bool = True, timeout_sec: int = 300, timeout_as_safe: bool = True, *args, **kwargs) -> Tensor:
        """
        :param task_name: id of the property, e.g., 1, 2, etc.
        """
        fname = self._next_fname()
        local_path = Path(self.fdir, fname)
        assert lb.size()[0] == 1 and ub.size()[0] == 1, 'ReluVal does not support multiple LB/UB bounds at once.'
        net.save(local_path, lb[0], ub[0])

        if self.objpath.is_file():
            # available in local machine
            cmds = self._local_cmds(local_path, task_name, collect_regions)
        else:
            # need to run via ssh, first copy the file to server
            tmp_name = 'tmp.nnet'
            try:
                subprocess.check_call(['scp', local_path, f'{self.ssh_user}@{self.ssh_ip}:~/{fname}'])
            except subprocess.CalledProcessError as e:
                print('Cannot even scp network file to server?', file=sys.stderr)
                raise e

            cmds = ['ssh', '-t', f'{self.ssh_user}@{self.ssh_ip}']
            cmds.extend(self._local_cmds(fname, task_name, collect_regions))

        def _run():
            if hide_err:
                return subprocess.check_output(cmds, timeout=timeout_sec, stderr=subprocess.DEVNULL)
            else:
                return subprocess.check_output(cmds, timeout=timeout_sec)

        try:
            output = _run()
        except subprocess.TimeoutExpired as e:
            if timeout_as_safe:
                # has run too long, just treated as no CEX found
                return torch.tensor([])
            else:
                raise e
        except subprocess.CalledProcessError as e:
            print('Error while calling ReluVal?!', file=sys.stderr)
            raise e

        outs = output.decode()  # somehow the returned string is binary, needs decode()
        cexs = self.extract(outs)
        data = [c.inputs for c in cexs]
        return torch.tensor(data)

    def _local_cmds(self, netfile: str, task_name, collect_regions: bool=False) -> List[str]:
        """ The cmds to run it in local machine.
        :param collect_regions: if True, the logs will also print out which splitted regions are safe or not
        """
        return [self.objpath, str(task_name), f'~/{netfile}', '0', '1' if collect_regions else '0', '0']

    @staticmethod
    def extract(logs: str) -> List[_CEX]:
        """ Extract Counterexamples returned in logs. """
        lines = logs.splitlines()

        def _is_verified():
            """ Quickly determine if it is fully determined. """
            for line in reversed(lines):
                if line.startswith('No adv!'):
                    return True
                elif line.startswith('adv found:'):
                    return False
            raise ValueError("Lines does not contain either 'No adv!' or 'adv found:'?!")

        def _parse_nums(nums_str: str) -> List[float]:
            assert nums_str.startswith('[') and nums_str.endswith(']')
            nums_str = nums_str[1:-1].strip()
            ns = nums_str.split(sep=' ')
            return [float(s) for s in ns]

        assert lines[1] == 'input ranges:'
        assert lines[4].startswith('check mode:')

        if _is_verified():
            return []

        curr = 5  # details starts here
        cexs = []
        while curr < len(lines):
            if lines[curr].startswith('['):
                # I was trying to print and collect their determined safe regions, branch deprecated.
                curr += 5
            elif lines[curr] == 'adv found:':
                # printing cex
                c = _CEX()

                input_line = lines[curr + 1]
                prefix = 'adv is: '
                assert input_line.startswith(prefix)
                assert input_line.endswith(']')
                c.inputs = _parse_nums(input_line[len(prefix):])
                c.inputs_normed = None

                output_line = lines[curr + 2]
                prefix = "it's output is: "
                assert output_line.startswith(prefix)
                assert output_line.endswith(']')
                c.outputs = _parse_nums(output_line[len(prefix):])
                c.outputs_normed = None

                cexs.append(c)
                curr += 3
            else:
                curr += 1
        return cexs
    pass
