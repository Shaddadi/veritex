""" Information needed for evaluation of ACAS Xu datasets. """

from __future__ import annotations

import datetime
import enum
import sys
from itertools import product
from pathlib import Path
from typing import List, Optional, Tuple, Iterable, Sequence, Union

import torch
from torch import Tensor, nn

from diffabs import AbsDom, AbsEle

sys.path.append(str(Path(__file__).resolve().parent.parent))

from art.prop import OneProp, AndProp
from art.utils import sample_points


ACAS_DIR = Path(__file__).resolve().parent.parent / 'data' / 'acas_nets'


class AcasIn(enum.IntEnum):
    RHO = 0
    THETA = 1
    PSI = 2
    V_OWN = 3
    V_INT = 4
    pass


class AcasOut(enum.IntEnum):
    CLEAR_OF_CONFLICT = 0
    WEAK_LEFT = 1
    WEAK_RIGHT = 2
    STRONG_LEFT = 3
    STRONG_RIGHT = 4
    pass


class AcasNetID(object):
    """ Indexing the provided networks in ACAS dataset. """
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y
        return

    def __hash__(self):
        return hash((self.x, self.y))

    def __eq__(self, other):
        if not isinstance(other, AcasNetID):
            return False
        return self.x == other.x and self.y == other.y

    def __repr__(self):
        return f'AcasNetID-{self.x}-{self.y}'

    def fpath(self, dir: str = ACAS_DIR):
        """ Return the corresponding file path for this network. """
        fname = f'ACASXU_run2a_{self.x}_{self.y}_batch_2000.nnet'
        if dir is not None:
            fname = Path(dir, fname)
        return fname

    def applicable_props(self, dom: AbsDom) -> List[AcasProp]:
        return [p for p in AcasProp.all_props(dom) if p.is_net_applicable(self)]

    # N_{x, y} is for indexing given networks
    XS = 5
    YS = 9

    @classmethod
    def all_ids(cls) -> List[AcasNetID]:
        """ Return a list of IDs for all provided networks in dataset. """
        return [AcasNetID(x, y) for x, y in product(range(1, cls.XS + 1), range(1, cls.YS + 1))]

    @classmethod
    def hard_ids(cls) -> List[AcasNetID]:
        """ Hand-picked harder networks (taking longer time and epochs than others) for demonstration. """
        ls = [
            (1, 1), (1, 9),
            (2, 7), (2, 9),
            (4, 1), (4, 5), (4, 8)
        ]
        return [AcasNetID(x, y) for x, y in ls]

    @classmethod
    def balanced_ids(cls) -> List[AcasNetID]:
        """ Hand-picked networks that are less imbalanced on output category samples.
            All others (except <1, 6>) may have < 0.1% samples for certain categories.
            Even <1, 6> has < 1% for certain category.
        """
        return [AcasNetID(1, y) for y in range(1, 6)]

    @classmethod
    def representative_ids(cls) -> List[AcasNetID]:
        """ First occurring networks for a new properties composition pattern. """
        ls = [
            (1, 1),  # prop 1, 3, 4, 5, 6a, 6b
            (1, 2),  # prop 1, 3, 4
            (1, 7),  # prop 1
            (1, 9),  # prop 1, 7
            (2, 1),  # prop 1, 2, 3, 4
            (2, 9),  # prop 1, 2, 3, 4, 8
            (3, 3),  # prop 1, 2, 3, 4, 9
            (4, 5)   # prop 1, 2, 3, 4, 10
        ]
        return [AcasNetID(x, y) for x, y in ls]

    @classmethod
    def casestudy_ids(cls) -> List[AcasNetID]:
        """ Case study requires to find a network violation instance for phi_2, here they are. """
        return [AcasNetID(2, y) for y in range(1, 6)]

    @classmethod
    def all_exp_ids(cls) -> List[AcasNetID]:
        """ One representative network as the representative of each of the 8 safety properties.
            The following are already the most balanced networks for each property, via inspect_net_props():
                <1,1>, <1,7>, <1,9>, <2,1>, <2,9>, <3,3>, <4,5>.
            While the following four networks for property 1,3,4 are both somewhat balanced:
                <1,2>, <1,3>, <1,4>, <1,5>.
            So I just run all of them as well.
        """
        ls = [
            # same as representative_ids() but with property-based ordering, also add one redundant network
            (1, 7),  # prop 1
            (2, 1),  # prop 1, 2, 3, 4
            (2, 9),  # prop 1, 2, 3, 4, 8
            (3, 3),  # prop 1, 2, 3, 4, 9
            (4, 5),  # prop 1, 2, 3, 4, 10
            (1, 2),  # prop 1, 3, 4
            (1, 1),  # prop 1, 3, 4, 5, 6a, 6b
            (1, 9),  # prop 1, 7
            (1, 3)  # another prop 1, 3, 4 for backup
        ]
        return [AcasNetID(x, y) for x, y in ls]

    @classmethod
    def goal_safety_ids(cls, dom: AbsDom) -> List[AcasNetID]:
        """ Those networks with safety violations (i.e., phi2 and phi8), to train to be safe by construction. """
        phi2 = AcasProp.property2(dom)
        phi8 = AcasProp.property8(dom)
        phi7 = AcasProp.property7(dom) # added by xiaodong yang for CAV'22 artifact

        ids = phi2.applicable.bool() | phi8.applicable.bool() | phi7.applicable.bool()
        ids = ids.nonzero(as_tuple=False)  # Batch x 2
        ids = [AcasNetID(row[0] + 1, row[1] + 1) for row in ids]
        return ids

    @classmethod
    def goal_accuracy_ids(cls, dom: AbsDom) -> List[AcasNetID]:
        """ Those networks checked safe are to show that the accuracy impact is mild. """
        phi2 = AcasProp.property2(dom)
        phi8 = AcasProp.property8(dom)

        ids = ~(phi2.applicable.bool() | phi8.applicable.bool())
        ids = ids.nonzero(as_tuple=False)  # Batch x 2
        ids = [AcasNetID(row[0] + 1, row[1] + 1) for row in ids]
        return ids
    pass


class AcasProp(OneProp):
    """ Defining a ACAS Xu safety property. """

    def __init__(self, name: str, dom: Optional[AbsDom], safe_fn: str, viol_fn: str, fn_args: Iterable):
        """
        :param safe_fn: function name to compute safety distance
        :param viol_fn: function name to compute violation distance
        :param fn_args: The arguments are shared between safe/viol functions
        """
        super().__init__(name, dom, safe_fn, viol_fn, fn_args)

        self.input_bounds = [
            (0.0, 60760.0),
            (-3.141593, 3.141593),
            (-3.141593, 3.141593),
            (100.0, 1200.0),
            (0.0, 1200.0)
        ]

        # for (de)normalization of outputs
        self.out_mean = 7.5188840201005975
        self.out_range = 373.94992

        # filter which nets as applicable
        self.applicable = torch.ones(AcasNetID.XS, AcasNetID.YS).byte()
        return

    def lbub(self, device=None) -> Tuple[Tensor, Tensor]:
        """ Return <LB, UB>, both of size <1xDim0>. """
        bs = torch.tensor(self.input_bounds)
        bs = bs.unsqueeze(dim=0)
        lb, ub = bs[..., 0], bs[..., 1]
        if device is not None:
            lb, ub = lb.to(device), ub.to(device)
        return lb, ub

    def set_all_applicable_as(self, apply: bool):
        fn = torch.ones_like if apply else torch.zeros_like
        self.applicable = fn(self.applicable)
        return

    def set_applicable(self, x: int, y: int, apply: bool):
        self.applicable[x - 1, y - 1] = int(apply)
        return

    def is_net_applicable(self, id: AcasNetID) -> bool:
        return bool(self.applicable[id.x - 1, id.y - 1])

    def set_input_bound(self, idx: int, new_low: float = None, new_high: float = None):
        low, high = self.input_bounds[idx]
        if new_low is not None:
            low = max(low, new_low)

        if new_high is not None:
            high = min(high, new_high)

        assert low <= high
        self.input_bounds[idx] = (low, high)
        return

    def applicable_net_paths(self, dir: str = ACAS_DIR) -> List[str]:
        """
        :param dir: directory prefix
        :return: all network names that this property is applicable to
        """
        ids = self.applicable.nonzero(as_tuple=False)  # Batch x 2
        ids = [AcasNetID(row[0] + 1, row[1] + 1) for row in ids]
        return [id.fpath(dir) for id in ids]

    def tex(self) -> str:
        """ 6a or 6b is also 6. """
        n = self.name.rsplit('property')[1]
        assert len(n) > 0
        if n.endswith('a') or n.endswith('b'):
            n = n[:-1]
        return '\\phi_{%s}' % n

    # ===== Below are predefined properties used in Reluplex paper's experiments. =====

    @classmethod
    def all_props(cls, dom: AbsDom) -> List[AcasProp]:
        names = [f'property{i}' for i in range(1, 6)]
        names.extend(['property6a', 'property6b'])
        names.extend([f'property{i}' for i in range(7, 11)])
        return [getattr(cls, n)(dom) for n in names]

    @classmethod
    def all_composed_props(cls, dom: AbsDom) -> List[AndProp]:
        """ These are the 8 set of props such that at least one network should satisfy. """
        def _fetch(ids: List[str]) -> AndProp:
            names = [f'property{i}' for i in ids]
            return AndProp([getattr(cls, n)(dom) for n in names])

        return [
            _fetch(['1']),
            _fetch(['1', '2', '3', '4']),
            _fetch(['1', '2', '3', '4', '8']),
            _fetch(['1', '2', '3', '4', '9']),
            _fetch(['1', '2', '3', '4', '10']),
            _fetch(['1', '3', '4']),
            _fetch(['1', '3', '4', '5', '6a', '6b']),
            _fetch(['1', '7'])
        ]

    @classmethod
    def property1(cls, dom: AbsDom):
        p = AcasProp(name='property1', dom=dom, safe_fn='col_le_val', viol_fn='col_ge_val',
                     fn_args=[AcasOut.CLEAR_OF_CONFLICT, 1500, 7.5188840201005975, 373.94992])  # mean/range hardcoded
        p.set_input_bound(AcasIn.RHO, new_low=55947.691)
        p.set_input_bound(AcasIn.V_OWN, new_low=1145)
        p.set_input_bound(AcasIn.V_INT, new_high=60)
        p.set_all_applicable_as(True)
        return p

    @classmethod
    def property2(cls, dom: AbsDom):
        p = AcasProp(name='property2', dom=dom, safe_fn='cols_not_max', viol_fn='cols_is_max',
                     fn_args=[AcasOut.CLEAR_OF_CONFLICT])
        p.set_input_bound(AcasIn.RHO, new_low=55947.691)
        p.set_input_bound(AcasIn.V_OWN, new_low=1145)
        p.set_input_bound(AcasIn.V_INT, new_high=60)
        p.set_all_applicable_as(True)
        for y in range(1, AcasNetID.YS + 1):
            p.set_applicable(1, y, False)

        p.set_applicable(3, 3, False) # added by xiaodong yang for CAV'22 artifact
        p.set_applicable(4, 2, False) # added by xiaodong yang for CAV'22 artifact
        return p

    @classmethod
    def property3(cls, dom: AbsDom):
        p = AcasProp(name='property3', dom=dom, safe_fn='cols_not_min', viol_fn='cols_is_min',
                     fn_args=[AcasOut.CLEAR_OF_CONFLICT])
        p.set_input_bound(AcasIn.RHO, new_low=1500, new_high=1800)
        p.set_input_bound(AcasIn.THETA, new_low=-0.06, new_high=0.06)
        p.set_input_bound(AcasIn.PSI, new_low=3.10)
        p.set_input_bound(AcasIn.V_OWN, new_low=980)
        p.set_input_bound(AcasIn.V_INT, new_low=960)
        p.set_all_applicable_as(True)
        for y in [7, 8, 9]:
            p.set_applicable(1, y, False)
        return p

    @classmethod
    def property4(cls, dom: AbsDom):
        p = AcasProp(name='property4', dom=dom, safe_fn='cols_not_min', viol_fn='cols_is_min',
                     fn_args=[AcasOut.CLEAR_OF_CONFLICT])
        p.set_input_bound(AcasIn.RHO, new_low=1500, new_high=1800)
        p.set_input_bound(AcasIn.THETA, new_low=-0.06, new_high=0.06)
        p.set_input_bound(AcasIn.PSI, new_low=-0.01, new_high=0.01)  # was [0, 0], for precise size, use [±0.01]
        p.set_input_bound(AcasIn.V_OWN, new_low=1000)
        p.set_input_bound(AcasIn.V_INT, new_low=700, new_high=800)
        p.set_all_applicable_as(True)
        for y in [7, 8, 9]:
            p.set_applicable(1, y, False)
        return p

    @classmethod
    def property5(cls, dom: AbsDom):
        p = AcasProp(name='property5', dom=dom, safe_fn='cols_is_min', viol_fn='cols_not_min',
                     fn_args=[AcasOut.STRONG_RIGHT])
        p.set_input_bound(AcasIn.RHO, new_low=250, new_high=400)
        p.set_input_bound(AcasIn.THETA, new_low=0.2, new_high=0.4)
        p.set_input_bound(AcasIn.PSI, new_low=-3.141592, new_high=-3.141592 + 0.005)
        p.set_input_bound(AcasIn.V_OWN, new_low=100, new_high=400)
        p.set_input_bound(AcasIn.V_INT, new_low=0, new_high=400)
        p.set_all_applicable_as(False)
        p.set_applicable(1, 1, True)
        return p

    @classmethod
    def property6a(cls, dom: AbsDom):
        p = AcasProp(name='property6a', dom=dom, safe_fn='cols_is_min', viol_fn='cols_not_min',
                     fn_args=[AcasOut.CLEAR_OF_CONFLICT])
        p.set_input_bound(AcasIn.RHO, new_low=12000, new_high=62000)
        p.set_input_bound(AcasIn.THETA, new_low=0.7, new_high=3.141592)
        p.set_input_bound(AcasIn.PSI, new_low=-3.141592, new_high=-3.141592 + 0.005)
        p.set_input_bound(AcasIn.V_OWN, new_low=100, new_high=1200)
        p.set_input_bound(AcasIn.V_INT, new_low=0, new_high=1200)
        p.set_all_applicable_as(False)
        p.set_applicable(1, 1, True)
        return p

    @classmethod
    def property6b(cls, dom: AbsDom):
        p = AcasProp(name='property6b', dom=dom, safe_fn='cols_is_min', viol_fn='cols_not_min',
                     fn_args=[AcasOut.CLEAR_OF_CONFLICT])
        p.set_input_bound(AcasIn.RHO, new_low=12000, new_high=62000)
        p.set_input_bound(AcasIn.THETA, new_low=-3.141592, new_high=-0.7)
        p.set_input_bound(AcasIn.PSI, new_low=-3.141592, new_high=-3.141592 + 0.005)
        p.set_input_bound(AcasIn.V_OWN, new_low=100, new_high=1200)
        p.set_input_bound(AcasIn.V_INT, new_low=0, new_high=1200)
        p.set_all_applicable_as(False)
        p.set_applicable(1, 1, True)
        return p

    @classmethod
    def property7(cls, dom: AbsDom):
        p = AcasProp(name='property7', dom=dom, safe_fn='cols_not_min', viol_fn='cols_is_min',
                     fn_args=[AcasOut.STRONG_LEFT, AcasOut.STRONG_RIGHT])
        p.set_input_bound(AcasIn.RHO, new_low=0, new_high=60760)
        p.set_input_bound(AcasIn.THETA, new_low=-3.141592, new_high=3.141592)
        p.set_input_bound(AcasIn.PSI, new_low=-3.141592, new_high=3.141592)
        p.set_input_bound(AcasIn.V_OWN, new_low=100, new_high=1200)
        p.set_input_bound(AcasIn.V_INT, new_low=0, new_high=1200)
        p.set_all_applicable_as(False)
        p.set_applicable(1, 9, True)
        return p

    @classmethod
    def property8(cls, dom: AbsDom):
        p = AcasProp(name='property8', dom=dom, safe_fn='cols_is_min', viol_fn='cols_not_min',
                     fn_args=[AcasOut.CLEAR_OF_CONFLICT, AcasOut.WEAK_LEFT])
        p.set_input_bound(AcasIn.RHO, new_low=0, new_high=60760)
        p.set_input_bound(AcasIn.THETA, new_low=-3.141592, new_high=0.75 * -3.141592)
        p.set_input_bound(AcasIn.PSI, new_low=-0.1, new_high=0.1)
        p.set_input_bound(AcasIn.V_OWN, new_low=600, new_high=1200)
        p.set_input_bound(AcasIn.V_INT, new_low=600, new_high=1200)
        p.set_all_applicable_as(False)
        p.set_applicable(2, 9, True)
        return p

    @classmethod
    def property9(cls, dom: AbsDom):
        p = AcasProp(name='property9', dom=dom, safe_fn='cols_is_min', viol_fn='cols_not_min',
                     fn_args=[AcasOut.STRONG_LEFT])
        p.set_input_bound(AcasIn.RHO, new_low=2000, new_high=7000)
        p.set_input_bound(AcasIn.THETA, new_low=-0.4, new_high=-0.14)
        p.set_input_bound(AcasIn.PSI, new_low=-3.141592, new_high=-3.141592 + 0.01)
        p.set_input_bound(AcasIn.V_OWN, new_low=100, new_high=150)
        p.set_input_bound(AcasIn.V_INT, new_low=0, new_high=150)
        p.set_all_applicable_as(False)
        p.set_applicable(3, 3, True)
        return p

    @classmethod
    def property10(cls, dom: AbsDom):
        p = AcasProp(name='property10', dom=dom, safe_fn='cols_is_min', viol_fn='cols_not_min',
                     fn_args=[AcasOut.CLEAR_OF_CONFLICT])
        p.set_input_bound(AcasIn.RHO, new_low=36000, new_high=60760)
        p.set_input_bound(AcasIn.THETA, new_low=0.7, new_high=3.141592)
        p.set_input_bound(AcasIn.PSI, new_low=-3.141592, new_high=-3.141592 + 0.01)
        p.set_input_bound(AcasIn.V_OWN, new_low=900, new_high=1200)
        p.set_input_bound(AcasIn.V_INT, new_low=600, new_high=1200)
        p.set_all_applicable_as(False)
        p.set_applicable(4, 5, True)
        return p
    pass


class _CommaString(object):
    """ A full string separated by commas. """
    def __init__(self, text: str):
        self.text = text
        return

    def __str__(self):
        return self.text

    def __repr__(self):
        return self.text

    def has_next_comma(self) -> bool:
        return ',' in self.text

    def _read_next(self) -> str:
        """
        :return: the raw string of next token before comma
        """
        if self.has_next_comma():
            token, self.text = self.text.split(',', maxsplit=1)
        else:
            token, self.text = self.text, ''
        return token.strip()

    def read_next_as_int(self) -> int:
        return int(self._read_next())

    def read_next_as_float(self) -> float:
        return float(self._read_next())

    def read_next_as_bool(self) -> bool:
        """ Parse the next token before comma as boolean, 1/0 for true/false. """
        num = self.read_next_as_int()
        assert num == 1 or num == 0, f'The should-be-bool number is {num}.'
        return bool(num)
    pass


class AcasNet(nn.Module):
    """ Compatible with the NNET format used in Reluplex. """

    def __init__(self, dom: AbsDom, input_size, output_size, hidden_sizes: List[int],
                 means: List[float] = None, ranges: List[float] = None):
        """
        :param means: of size input_size + 1, one extra for the output
        :param ranges: of size input_size + 1, one extra for the output
        """
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.n_layers = len(hidden_sizes) + 1

        # By default, assume normalized, therefore (d - mean) / range doesn't change the value.
        # One more element for output data mean/range.
        self.means = means if means is not None else [0.0] * (self.input_size + 1)
        self.ranges = ranges if ranges is not None else [1.0] * (self.input_size + 1)

        self.acti = dom.ReLU()
        self.all_linears = nn.ModuleList()
        in_sizes = [self.input_size] + self.hidden_sizes
        out_sizes = self.hidden_sizes + [self.output_size]
        for in_size, out_size in zip(in_sizes, out_sizes):
            self.all_linears.append(dom.Linear(in_size, out_size))
        return

    def __str__(self):
        """ Just print everything for information. """
        ss = [
            '--- AcasNet ---',
            'Num layers: %d (i.e. hidden + output, excluding input layer)' % self.n_layers,
            'Input size: %d' % self.input_size,
            'Hidden sizes (len %d): ' % len(self.hidden_sizes) + str(self.hidden_sizes),
            'Output size: %d' % self.output_size,
            'Means for scaling (len %d): ' % len(self.means) + str(self.means),
            'Ranges for scaling (len %d): ' % len(self.ranges) + str(self.ranges),
            'Activation: %s' % self.acti,
            '--- End of AcasNet ---'
        ]
        return '\n'.join(ss)

    def normalize_inputs(self, t: Tensor, mins: Sequence[float], maxs: Sequence[float]) -> Tensor:
        """ Normalize: ([min, max] - mean) / range """
        slices = []
        for i in range(self.input_size):
            slice = t[:, i:i+1]
            slice = slice.clamp(mins[i], maxs[i])
            slice -= self.means[i]
            slice /= self.ranges[i]
            slices.append(slice)
        return torch.cat(slices, dim=-1)

    def denormalize_outputs(self, t: Tensor) -> Tensor:
        """ Denormalize: v * range + mean """
        # In NNET files, the mean/range of output is stored in [-1] of array.
        # All are with the same mean/range, so I don't need to slice.
        t *= self.ranges[-1]
        t += self.means[-1]
        return t

    def forward(self, x: Union[Tensor, AbsEle]) -> Union[Tensor, AbsEle]:
        """ Normalization and Denomalization are called outside this method. """
        for lin in self.all_linears[:-1]:
            x = lin(x)
            x = self.acti(x)

        x = self.all_linears[-1](x)
        return x

    def reset_params(self, strategy: str = None):
        """ Reset all parameters to some initial state. """
        if strategy is None:
            return
        elif strategy == 'default':
            for lin in self.all_linears:
                lin.reset_parameters()
        else:
            raise NotImplementedError()
        return

    def save_nnet(self, outpath: str, mins: Iterable[float], maxs: Iterable[float]):
        """ Output the current parameters to a file, in the same format as NNET files.
            Following the code (already validated) in NNET.
        :param mins: lower bounds for input vector
        :param maxs: upper bounds for input vector
        """
        with open(outpath, 'w') as f:
            # headers
            timestr = datetime.datetime.now().strftime('%m/%d/%Y %H:%M:%S')
            f.writelines([
                '// The contents of this file are licensed under the Creative Commons\n',
                '// Attribution 4.0 International License: https://creativecommons.org/licenses/by/4.0/\n',
                '// Neural Network File Format by Kyle Julian, Stanford 2016 (generated on %s)\n' % timestr
            ])

            def _write_comma_line(vs):
                """ Write a list of values into file ending with \n, each one followed by a comma.
                :param vs: a list of values
                """
                if isinstance(vs, Tensor):
                    # otherwise, enumeration will output strings like tensor(1.0)
                    vs = vs.numpy()

                for v in vs:
                    f.write(str(v) + ',')
                f.write('\n')
                return

            # line 1 - basics
            max_hidden = 0 if len(self.hidden_sizes) == 0 else max(self.hidden_sizes)
            max_layer_size = max(self.input_size, max_hidden, self.output_size)
            _write_comma_line([self.n_layers, self.input_size, self.output_size, max_layer_size])

            # line 2 - layer sizes
            layer_sizes = [self.input_size] + self.hidden_sizes + [self.output_size]
            _write_comma_line(layer_sizes)

            # line 3 - symmetric
            f.write('0,\n')

            # line 4 - mins of input
            _write_comma_line(mins)

            # line 5 - maxs of input
            _write_comma_line(maxs)

            # line 6 - means
            _write_comma_line(self.means)

            # line 7 - ranges
            _write_comma_line(self.ranges)

            # writing parameters
            for linear in self.all_linears:
                in_size = linear.in_features
                out_size = linear.out_features

                # (1) write "weights"
                w = linear.weight.data
                for i in range(out_size):
                    for j in range(in_size):
                        f.write('%e,' % w[i][j])
                    f.write('\n')

                # (2) write "biases"
                b = linear.bias.data
                for i in range(out_size):
                    # only 1 item for each
                    f.write('%e,\n' % b[i])
        return

    @classmethod
    def load_nnet(cls, filepath: str, dom: AbsDom, device=None) -> Tuple[AcasNet, List[float], List[float]]:
        """ Load from dumped file in NNET format.
        :return: Tuple of <AcasNet, input mins vector, input maxs vector>
        """
        # ===== Basic Initializations =====
        _num_layers = 0  # Number of layers in the network (excluding inputs, hidden + output = num_layers).
        _input_size = 0  # Number of inputs to the network.
        _output_size = 0  # Number of outputs to the network.
        _max_layer_size = 0  # Maximum size dimension of a layer in the network.

        _layer_sizes = []  # Array of the dimensions of the layers in the network.

        _symmetric = False  # Network is symmetric or not. (was 1/0 for true/false)

        _mins = []  # Minimum value of inputs.
        _maxs = []  # Maximum value of inputs.

        _means = []  # Array of the means used to scale the inputs and outputs.
        _ranges = []  # Array of the ranges used to scale the inputs and outputs.

        _layer_weights = []  # holding concrete weights of each layer
        _layer_biases = []  # holding concrete biases of each layer

        # ===== Now loading from files =====
        if not Path(filepath).is_file():
            raise FileNotFoundError(f'{filepath} is not a valid path for NNET file.')

        with open(filepath, 'r') as f:
            line = f.readline()
            while line.startswith('//'):
                # ignore first several comment lines
                line = f.readline()

            # === Line 1: Basics ===
            data = _CommaString(line)
            _num_layers = data.read_next_as_int()
            _input_size = data.read_next_as_int()
            _output_size = data.read_next_as_int()
            _max_layer_size = data.read_next_as_int()

            # === Line 2: Layer sizes ===
            data = _CommaString(f.readline())
            for _ in range(_num_layers + 1):
                _layer_sizes.append(data.read_next_as_int())

            assert _layer_sizes[0] == _input_size
            assert _layer_sizes[-1] == _output_size
            assert all(size <= _max_layer_size for size in _layer_sizes)
            assert len(_layer_sizes) >= 2, f'Loaded layer sizes have {len(_layer_sizes)} (< 2) elements?! Too few.'

            # === Line 3: Symmetric ===
            data = _CommaString(f.readline())
            _symmetric = data.read_next_as_bool()
            assert _symmetric is False, "We don't know what symmetric==True means."

            # It has to read by line, because in following lines, I noticed some files having more values than needed..

            # === Line 4: Mins of input ===
            data = _CommaString(f.readline())
            for _ in range(_input_size):
                _mins.append(data.read_next_as_float())

            # === Line 5: Maxs of input ===
            data = _CommaString(f.readline())
            for _ in range(_input_size):
                _maxs.append(data.read_next_as_float())

            # === Line 6: Means ===
            data = _CommaString(f.readline())
            # the [-1] is storing the size for output normalization
            for _ in range(_input_size + 1):
                _means.append(data.read_next_as_float())

            # === Line 7: Ranges ===
            data = _CommaString(f.readline())
            # the [-1] is storing the size for output normalization
            for _ in range(_input_size + 1):
                _ranges.append(data.read_next_as_float())

            # === The rest are layer weights/biases. ===
            for k in range(_num_layers):
                in_size = _layer_sizes[k]
                out_size = _layer_sizes[k + 1]

                # read "weights"
                tmp = []
                for i in range(out_size):
                    row = []
                    data = _CommaString(f.readline())
                    for j in range(in_size):
                        row.append(data.read_next_as_float())
                    tmp.append(row)
                    assert not data.has_next_comma()

                """ To fully comply with NNET in Reluplex, DoubleTensor is necessary.
                    Otherwise it may record 0.613717 as 0.6137170195579529.
                    But to make everything easy in PyTorch, I am just using FloatTensor.
                """
                _layer_weights.append(torch.tensor(tmp))

                # read "biases"
                tmp = []
                for i in range(out_size):
                    # only 1 item for each
                    data = _CommaString(f.readline())
                    tmp.append(data.read_next_as_float())
                    assert not data.has_next_comma()

                _layer_biases.append(torch.tensor(tmp))
                pass

            data = _CommaString(f.read())
            assert not data.has_next_comma()  # should have no more data

        # ===== Use the parsed information to build AcasNet =====
        _hidden_sizes = _layer_sizes[1:-1]  # exclude inputs and outputs sizes
        net = AcasNet(dom, _input_size, _output_size, _hidden_sizes, _means, _ranges)

        # === populate weights and biases ===
        assert len(net.all_linears) == len(_layer_weights) == len(_layer_biases)
        for i, linear in enumerate(net.all_linears):
            linear.weight.data = _layer_weights[i]
            linear.bias.data = _layer_biases[i]

        if device is not None:
            net = net.to(device)
        return net, _mins, _maxs

    @classmethod
    def compare_params(cls, net1, net2) -> bool:
        """ Compare the weights differences for two concrete AcasNet. """
        assert isinstance(net1, AcasNet) and isinstance(net2, AcasNet)
        assert len(net1.all_linears) == len(net2.all_linears)

        def _cmp(t1, t2):
            diff = t2 - t1
            diff_abs = diff.abs()
            print('Diff: [max] %f, [min] %f' % (diff_abs.max(), diff_abs.min()))
            print(diff)

            factor = t2 / t1
            factor_abs = factor.abs()
            print('Factor: [max] %f, [min] %f' % (factor_abs.max(), factor_abs.min()))
            print(factor)
            return

        all_eq = True
        for i in range(len(net1.all_linears)):
            linear1 = net1.all_linears[i]
            linear2 = net2.all_linears[i]
            assert type(linear1) is type(linear2)

            ws1 = linear1.weight.data
            bs1 = linear1.bias.data
            ws2 = linear2.weight.data
            bs2 = linear2.bias.data

            if not torch.equal(ws1, ws2):
                all_eq = False
                print('Linear', i, 'ws diffs are:')
                _cmp(ws1, ws2)

            if not torch.equal(bs1, bs2):
                all_eq = False
                print('Linear', i, 'bs diffs are:')
                _cmp(bs1, bs2)
        return all_eq
    pass


# ===== Below are some inspection methods =====


def inspect_net_props(dom: AbsDom):
    """ Inspect the properties each network should satisfy.
        The network ids are grouped by all property sets, so as to pick the most balanced ones among them.
    """
    unique_ids = []
    grouped = {}
    for nid in AcasNetID.all_ids():
        props = AndProp(nid.applicable_props(dom))
        # print(f'{nid.x}, {nid.y}', end='\t')

        prop_ids = [p.name.split('property')[1] for p in props.props]
        if prop_ids not in unique_ids:
            unique_ids.append(prop_ids)
            grouped[props.name] = [nid]
        else:
            grouped[props.name].append(nid)
        print(f'{nid}: prop', ','.join(prop_ids))

    print('Unique prop ids are:', len(unique_ids))
    for ids in sorted(unique_ids):
        print(ids)
    print()
    print('Grouped for all props:')
    for k, v in grouped.items():
        print('===== Props:', k, '=====')
        print('Nets:')
        for nid in v:
            print(nid)
    return


def inspect_prop_pts(dom: AbsDom, sample_size=1000):
    """ Sample points for each property, and watch for any safe≠0 points.

                Pt Safe Ratio
    Property 2	0.99
    Property 8	0.999

    All the rest are with 1.0 safe ratio, these are are perhaps due to numerical error?
    """
    def debug_unsafe_point(prop: AcasProp):
        fpath = prop.applicable_net_paths()[0]
        print('Picked nnet file:', fpath)

        net, bound_mins, bound_maxs = AcasNet.load_nnet(fpath, dom)
        lb, ub = prop.lbub()

        net = net.to(device)
        lb, ub = lb.to(device), ub.to(device)

        lb = net.normalize_inputs(lb, bound_mins, bound_maxs)
        ub = net.normalize_inputs(ub, bound_mins, bound_maxs)

        for i in range(20):  # run 20 times
            sampled_pts = sample_points(lb, ub, sample_size)
            with torch.no_grad():
                pt_outs = net(sampled_pts)
                pt_outs = dom.Ele.by_intvl(pt_outs, pt_outs)
                safe_dist = prop.safe_dist(pt_outs)

            safe_bits = safe_dist == 0.
            safe_ratio = len(safe_bits.nonzero(as_tuple=False)) / float(sample_size)
            print('Iter', i, ': Safe ratio for point outputs:', safe_ratio)
            if safe_ratio != 1.0:
                spurious_bits = ~ safe_bits
                spurious_dist = safe_dist[spurious_bits]
                print('\t', f'spurious dists: [ {spurious_dist.min()} ~ {spurious_dist.max()} ]')
                print('\t', 'spurious dists:', spurious_dist)
                pass
        return

    for p in AcasProp.all_props(dom):
        print(f'===== For {p.name} =====')
        debug_unsafe_point(p)
        print('\n')
    return


if __name__ == '__main__':
    from diffabs import DeeppolyDom
    dom = DeeppolyDom()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    inspect_net_props(dom)
    # inspect_prop_pts(dom)
    pass
