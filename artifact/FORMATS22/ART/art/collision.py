""" Information needed for evaluation of Collision Avoidance dataset from Ehlers 2017.

    Facts about the provided safety properties in Collision Avoidance dataset:
    *   In each original rlv file, input bounds and output constraints are specified.
        While in the adapted nnet files, the last layer is converted to 1 neuron,
        this is fine for verification, but forbids further training.
    *   Similarly, all network parameters in original rlv files are the same,
        only properties differ. While in the adapted nnet files, the network weights
        are different due to verification oriented adaptation.
    *   The 500 provided properties are essentially safety margins of 1st~100th (0-based)
        dataset points, specifying different epsilon values among different properties.
        Each central point has exactly 5 different epsilons, which means they could be
        subsumed by larger epsilons.
    *   For each property, the same epsilon value applies to all dimensions.
    *   The file name _UNSAT means no safety margin violation is found, vice versa for _SAT.
        There are 172/500 violation (SAT) cases and 328/500 safe (UNSAT) cases.
        Our experiment ensures that after training, all are safe (UNSAT).
    *   Also, even if the prediction is incorrect (4/3000), the safety property still
        specifies according to the true label but not prediction, because it's safety margin.
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Union, List, Tuple, Callable, Optional

import torch
from torch import Tensor, nn

from diffabs import AbsDom, AbsEle
from diffabs.utils import valid_lb_ub

sys.path.append(str(Path(__file__).resolve().parent.parent))

from art.prop import OneProp
from art.exp import ConcIns


COLLISION_DIR = Path(__file__).resolve().parent.parent / 'data' / 'collision'

IN_FEATURES = 6
OUT_FEATURES = 2


class CollisionProp(OneProp):
    """ Defining the Collision Avoidance/Detection safety property. """
    def __init__(self, dom: Optional[AbsDom],
                 bound_mins: Union[Tensor, List[float]], bound_maxs: Union[Tensor, List[float]], larger_category: int):
        """
        :param bound_mins: input lower bounds
        :param bound_maxs: input upper bounds
        :param larger_category: the output category that is supposed to be larger
        """
        if not isinstance(bound_mins, Tensor):
            bound_mins = torch.tensor(bound_mins)
        if not isinstance(bound_maxs, Tensor):
            bound_maxs = torch.tensor(bound_maxs)

        valid_lb_ub(bound_mins, bound_maxs)
        self.bound_mins = bound_mins
        self.bound_maxs = bound_maxs
        self.larger_category = larger_category

        ''' In planet and marabou they try to find cex with y0 <= 0,
            which means safety prop is y0 >= 0 (accept y0=0 as well).
        '''
        super().__init__('CollisionProp', dom,
                         safe_fn='cols_is_max', viol_fn='cols_not_max', fn_args=(self.larger_category,))
        return

    def epsilon(self) -> float:
        eps = (self.bound_maxs - self.bound_mins) / 2.
        avg_eps = eps.mean().item()
        assert torch.allclose(eps, torch.full_like(eps, avg_eps))  # same eps on every dimension
        return avg_eps

    def lbub(self, device=None) -> Tuple[Tensor, Tensor]:
        """ Return <LB, UB>, both of size <1xDim0>. """
        lb = self.bound_mins.unsqueeze(dim=0)
        ub = self.bound_maxs.unsqueeze(dim=0)
        if device is not None:
            lb, ub = lb.to(device), ub.to(device)
        return lb, ub

    @classmethod
    def load(cls, fpath, dom: Optional[AbsDom]) -> CollisionProp:
        """ Separate the loading of network and prop because all the networks are the same, no need to load many times.
            Now that the provided robustness properties are according to true labels from dataset but not predictions,
            we can just use these properties directly.
        """
        assert Path(fpath).is_file(), f'{fpath} not found for the collision avoidance correctness property'
        f = open(fpath, 'r')

        def getline() -> List[str]:
            tokens = f.readline().strip().split(' ')
            return [t.strip() for t in tokens]

        line = getline()  # the first line to start with
        input_vars, line = _parse_input_vars(line, getline)
        assert len(input_vars) == IN_FEATURES

        while line[0] != 'Assert':
            line = getline()

        # input bounds
        lbs = torch.zeros(len(input_vars))
        ubs = torch.zeros(len(input_vars))
        for _ in range(len(input_vars)):
            assert line[0] == 'Assert' and line[1] == '<='
            lb = float(line[2])
            assert line[3] == '1.0'
            var = line[4]
            line = getline()
            assert line[0] == 'Assert' and line[1] == '>='
            ub = float(line[2])
            assert line[3] == '1.0'
            assert line[4] == var
            line = getline()

            idx = input_vars.index(var)
            lbs[idx] = lb
            ubs[idx] = ub

        # output constraints
        assert line[0] == 'Assert' and line[1] == '>=' and line[2] == '0.0' and line[3] == '-1.0' and line[5] == '1.0'
        # note that original rlv file looks for violation, so our safety prop should take the opposite
        larger_category = int(line[6][-1])

        line = getline()
        assert len(line) == 0 or line[0] == ''  # done

        f.close()
        return CollisionProp(dom, lbs, ubs, larger_category)
    pass


def _parse_input_vars(line: List[str], getline_fn: Callable[[], List[str]]) -> Tuple[List[str], List[str]]:
    """ Given the current line, continue to parse all input names.
    :param line: the split line to start
    :param getline_fn: call to get the next line from file
    :return: the parsed input variable names, and the next line to parse subsequently
    """
    _input_vars = []
    while True:
        if len(line) != 2:
            break

        assert line[0] == 'Input'
        _input_vars.append(line[1])
        line = getline_fn()
    return _input_vars, line


def cluster_props(all_props: List[CollisionProp]) -> dict:
    """ Many provided properties are overlapping -- some subsumes some others, cluster them for later usage.
    :return: a dict of central points -> sorted props in decreasing order of their epsilons
    """
    d = defaultdict(list)  # base point -> List of props

    def _locate_similar(base: Tensor):
        """ If there is already some very close base points in the dict, return that as the key.
            Otherwise, return the new point as key.
        """
        for k in d.keys():
            if torch.allclose(k, base):
                return k
        return base

    for prop in all_props:
        mid = (prop.bound_mins + prop.bound_maxs) / 2.
        base = _locate_similar(mid)
        d[base].append(prop)

    # validate that all props assigned to the same central point has the same labels
    for props in d.values():
        assert len(props) == 5
        categories = [p.larger_category for p in props]
        assert ([c == categories[0] for c in categories])

    for k in d.keys():
        # sort in decreasing order according to epsilon
        d[k] = sorted(d[k], key=lambda p: p.epsilon(), reverse=True)
    return d


# hardcode the architecture in dataset
FC1_NEURONS = 40
MP_KERNEL_SIZE = 4
MP_STRIDE = 2
FC2_NEURONS = 19


class CollisionMPNet(nn.Module):
    """ The network provided in the dataset. """

    def __init__(self, dom):
        super().__init__()

        self.fc1 = dom.Linear(IN_FEATURES, FC1_NEURONS)
        self.relu = dom.ReLU()
        self.maxpool = dom.MaxPool1d(kernel_size=MP_KERNEL_SIZE, stride=MP_STRIDE)
        self.fc2 = dom.Linear(FC2_NEURONS, FC2_NEURONS)
        self.fc3 = dom.Linear(FC2_NEURONS, OUT_FEATURES)
        return

    def __iter__(self):
        return iter([self.fc1, self.relu, self.maxpool, self.fc2, self.relu, self.fc3])

    def forward(self, x: Union[AbsEle, Tensor]):
        x = self.relu(self.fc1(x))

        x = x.unsqueeze(dim=1)
        x = self.maxpool(x)
        x = x.squeeze(dim=1)

        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def reset_params(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        self.fc3.reset_parameters()
        return

    @classmethod
    def load(cls, fpath, dom: AbsDom, device=None) -> CollisionMPNet:
        """ Now that the network architectures are the same, we hardcode much of the loading procedure. """
        assert Path(fpath).is_file(), f'{fpath} not found for the collision avoidance network'
        f = open(fpath, 'r')

        def getline() -> List[str]:
            tokens = f.readline().strip().split(' ')
            return [t.strip() for t in tokens]

        line = getline()  # the first line to start with
        net = cls(dom)

        input_vars, line = _parse_input_vars(line, getline)
        assert len(input_vars) == IN_FEATURES

        def parse_fc_layer(line: List[str], last_layer_vars: List[str],
                           nrows_limit: int = -1) -> Tuple[List[str], Tensor, Tensor, str, List[str]]:
            """
            :return: this layer variable names, this layer weights, this layer biases, activation function, next line
            """
            _layer_acti = line[0]
            _layer_vars, _layer_ws, _layer_bs = [], [], []
            cnt = 0
            while True:
                if line[0] != _layer_acti:
                    break

                if nrows_limit > 0 and cnt >= nrows_limit:
                    # stop when having read this many lines
                    break

                cnt += 1
                _layer_vars.append(line[1])
                _layer_bs.append(float(line[2]))

                ws_line = line[3:]
                assert len(ws_line) == 2 * len(last_layer_vars), 'remaining list should have w~var paired'

                w_row = torch.zeros(len(last_layer_vars))
                for i in range(len(ws_line) // 2):
                    v, var = float(ws_line[2 * i]), ws_line[2 * i + 1]
                    idx = last_layer_vars.index(var)
                    w_row[idx] = v
                _layer_ws.append(w_row)
                line = getline()

            _layer_ws = torch.stack(_layer_ws, dim=0)
            _layer_bs = torch.tensor(_layer_bs)
            return _layer_vars, _layer_ws, _layer_bs, _layer_acti, line

        fc1_vars, fc1_ws, fc1_bs, fc1_acti, line = parse_fc_layer(line, input_vars)
        assert len(fc1_vars) == FC1_NEURONS
        net.fc1.weight.data = fc1_ws
        net.fc1.bias.data = fc1_bs
        assert fc1_acti == 'ReLU'

        def parse_maxpool1d(line: List[str], last_layer_vars) -> Tuple[List[str], int, int, List[str]]:
            """
            :return: this layer variable names, kernel size, stride size, next line to parse
            """
            _layer_vars = []
            _kernel_sizes = []
            while True:
                if line[0] != 'MaxPool':
                    break

                _layer_vars.append(line[1])

                window_vars = line[2:]
                _kernel_sizes.append(len(window_vars))
                # now that all networks are the same architecture, we just hardcode the maxpool structure

                line = getline()

            assert all([_kernel_sizes[0] == _kernel_sizes[i] for i in range(len(_kernel_sizes))])
            # hardcode stride = 2
            return _layer_vars, _kernel_sizes[0], 2, line

        mp_vars, mp_kernel, mp_stride, line = parse_maxpool1d(line, fc1_vars)
        assert len(mp_vars) == FC2_NEURONS and mp_kernel == MP_KERNEL_SIZE and mp_stride == MP_STRIDE

        # 2nd fc layer, after maxpool
        fc2_vars, fc2_ws, fc2_bs, fc2_acti, line = parse_fc_layer(line, mp_vars)
        assert len(fc2_vars) == FC2_NEURONS
        net.fc2.weight.data = fc2_ws
        net.fc2.bias.data = fc2_bs
        assert fc2_acti == 'ReLU'

        # last fc layer, just before final output
        fc3_vars, fc3_ws, fc3_bs, fc3_acti, line = parse_fc_layer(line, fc2_vars, nrows_limit=2)
        assert len(fc3_vars) == OUT_FEATURES
        net.fc3.weight.data = fc3_ws
        net.fc3.bias.data = fc3_bs
        assert fc3_acti == 'Linear'

        assert line[0] == 'Linear' and line[1] == 'outX0' and line[2] == '0.0' and line[3] == '1.0'
        line = getline()
        assert line[0] == 'Linear' and line[1] == 'outX1' and line[2] == '0.0' and line[3] == '1.0'
        line = getline()

        # the rest are for correctness property input bounds and output constraints, done
        assert line[0] == 'Assert'

        f.close()
        if device is not None:
            net = net.to(device)
        return net
    pass


class CollisionData(ConcIns):
    """ Storing the provided concrete data points for collision dataset. Loads to CPU/GPU automatically. """
    @classmethod
    def load(cls, device):
        """ Note that in Ehlers 2017, they only aim to fit all 3k prepared tuples, thus there is no test set.
            Even with that, they can only have 86/100 tries get 100% accuracy and out of the 500 network instances,
            they have 172/500 to be unsafe.
        """
        fpath = Path(COLLISION_DIR, 'collisions.csv')
        inputs, labels = [], []
        with open(fpath, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for line in reader:
                input = [float(s) for s in line[:-1]]
                inputs.append(input)

                label = int(line[-1])
                assert label in [0, 1]
                labels.append(label)

        inputs = torch.tensor(inputs).to(device)
        labels = torch.tensor(labels).long().to(device)
        return cls(inputs, labels)
    pass
