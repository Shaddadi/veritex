""" Basic information for NN verification evaluations. The detailed results of other tools come from VNN-COMP 2020
    report draft avaiable at https://sites.google.com/view/vnn20/vnncomp.
"""

import sys
from argparse import Namespace
from pathlib import Path
from typing import List, Tuple

sys.path.append(str(Path(__file__).resolve().parent.parent))

from art.acas import AcasNetID


class VNN20Info(object):
    def __init__(self):
        self.root_dir = Path(__file__).resolve().parent.parent / 'data' / 'vnn-comp2020'
        self.tools = ['nnenum', 'NNV', 'PeregriNN', 'MIPVerify', 'venus', 'eran']

        self.prefix_all = 'acasxu-all'
        self.prefix_hard = 'acasxu-hard'
        self.results_all = self.init(True)
        self.results_hard = self.init(False)
        return

    def init(self, for_all: bool) -> Namespace:
        """
        :param for_all: all or hard
        """
        prefix = self.prefix_all if for_all else self.prefix_hard
        props, nets = self.load(self.root_dir / (prefix + '-front.txt'), False)
        labels, = self.load(self.root_dir / (prefix + '-results.txt'), False)
        assert len(props) == len(nets) == len(labels)
        total_rows = len(props)

        tool_times, tool_answers = [], []
        for tool in self.tools:
            keep_first_line = tool in ['NNV', 'PeregriNN', 'venus']
            ret = self.load(self.root_dir / (prefix + f'-{tool}.txt'), keep_first_line)
            if len(ret) == 1:
                # Some tools only report the time, without SAT/UNSAT answer.
                tool_times.append(ret[0])
                tool_answers.append([None] * len(ret[0]))
            else:
                # Some tools report both time, in which [0] is time, [1] is answer.
                # There may be [2] [3], as in MIPVerify, it doesn't matter.
                tool_times.append(ret[0])
                tool_answers.append(ret[1])
            assert len(tool_times[-1]) == len(tool_answers[-1]) == total_rows

        d = Namespace()
        d.props = props
        d.nets = nets
        d.labels = labels
        d.tool_times = tool_times
        d.tool_answers = tool_answers
        return d

    def load(self, fpath: Path, keep_first_line: bool) -> List[List]:
        """ Load columns of the results table stored in a specific file. """
        with open(fpath, 'r') as f:
            lines = f.readlines()

        if not keep_first_line:
            lines = lines[1:]  # first line is header
        lines = [line.strip() for line in lines]
        lines = [line for line in lines if len(line) > 0]  # e.g., NNV txt may have some empty lines at the end
        items = [line.split() for line in lines]  # split() to handle both space and tab

        # unzip into multiple columns
        width = len(items[0])
        assert width > 0

        ret = []
        for i in range(width):
            subi = [p[i] for p in items]
            ret.append(subi)
        assert all([len(ls) == len(ret[0]) for ls in ret])  # all returned lists have the same length
        return ret

    def pp(self, for_all: bool) -> str:
        """ Pretty print. """
        d = self.results_all if for_all else self.results_hard
        lines = []
        lines.append('\t'.join(['Prop', 'Net', 'Result'] + self.tools))
        lines.append('----------------------------------------')
        cols = [d.props, d.nets, d.labels] + d.tool_times
        for i in range(len(d.props)):
            line = [ls[i] for ls in cols]
            lines.append('\t'.join(line))
        s = '\n'.join(lines)
        print(s)
        return s

    def query(self, prop: int, net_id: AcasNetID, for_all: bool) -> Tuple[str, List, List]:
        """
        :return: label, times list, answers list
                 the tools list is not returned, can be accessed by self.tools
        """
        d = self.results_all if for_all else self.results_hard
        net_id_str = f'{net_id.x}-{net_id.y}'
        for i in range(len(d.props)):
            if int(d.props[i]) == prop and d.nets[i] == net_id_str:
                times = [ls[i] for ls in d.tool_times]
                answers = [ls[i] for ls in d.tool_answers]
                return d.labels[i], times, answers
        raise IndexError(f'Record not found for prop {prop} and net {net_id}.')
    pass


if __name__ == '__main__':
    info = VNN20Info()
    print(info.root_dir)
    info.pp(True)
    info.pp(False)
    pass
