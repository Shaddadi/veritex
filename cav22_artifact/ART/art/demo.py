""" Motivating example in paper. """

from __future__ import annotations

import enum
import itertools
import sys
from math import pi, ceil
from pathlib import Path
from timeit import default_timer as timer
from typing import List, Tuple, Optional, Union

import torch
from torch import Tensor, nn
from torch.optim import Adam
from torch.utils import data

from diffabs import AbsDom, AbsEle, IntervalDom, AbsData

sys.path.append(str(Path(__file__).resolve().parent.parent))

from art.prop import OneProp
from art.bisecter import Bisecter
from art import utils


class DemoIn(enum.IntEnum):
    V = 0
    THETA = 1
    pass


class DemoOut(enum.IntEnum):
    HIGHLIGHT = 0
    IGNORE = 1
    pass


class DemoProp(OneProp):
    def __init__(self, name: str, dom: Optional[AbsDom], safe_fn: str, viol_fn: str, fn_args: List):
        """
        :param safe_fn: function name to compute safety distance
        :param viol_fn: function name to compute violation distance
        :param fn_args: The arguments are shared between safe/viol functions
        """
        super().__init__(name, dom, safe_fn, viol_fn, fn_args)

        self.input_bounds = [
            (-5, 5),  # v
            (-pi, pi)  # theta
        ]
        return

    def lbub(self, device=None) -> Tuple[Tensor, Tensor]:
        """ Return <LB, UB>, both of size <1xDim0>. """
        bs = torch.tensor(self.input_bounds)
        bs = bs.unsqueeze(dim=0)
        lb, ub = bs[..., 0], bs[..., 1]
        if device is not None:
            lb, ub = lb.to(device), ub.to(device)
        return lb, ub

    def set_input_bound(self, idx: int, new_low: float = None, new_high: float = None):
        low, high = self.input_bounds[idx]
        if new_low is not None:
            low = max(low, new_low)

        if new_high is not None:
            high = min(high, new_high)

        assert low <= high
        self.input_bounds[idx] = (low, high)
        return

    # ===== Below are predefined properties used in demo. =====

    @classmethod
    def all_props(cls, dom: AbsDom) -> List[DemoProp]:
        return [cls.property1(dom), cls.property2(dom)]

    @classmethod
    def property1(cls, dom: AbsDom) -> DemoProp:
        p = DemoProp(name='phi1', dom=dom, safe_fn='cols_is_max', viol_fn='cols_not_max',
                     fn_args=[DemoOut.HIGHLIGHT])
        p.set_input_bound(DemoIn.V, new_low=4.)
        return p

    @classmethod
    def property2(cls, dom: AbsDom) -> DemoProp:
        p = DemoProp(name='phi2', dom=dom, safe_fn='cols_is_max', viol_fn='cols_not_max',
                     fn_args=[DemoOut.HIGHLIGHT])
        p.set_input_bound(DemoIn.V, new_low=0.)
        p.set_input_bound(DemoIn.THETA, new_low=pi/4, new_high=3*pi/4)
        return p

    @classmethod
    def property12(cls, dom: AbsDom) -> DemoProp:
        """ Combining the above two safety properties. """
        p = DemoProp(name='phi12', dom=dom, safe_fn='cols_is_max', viol_fn='cols_not_max',
                     fn_args=[DemoOut.HIGHLIGHT])
        p.set_input_bound(DemoIn.V, new_low=0.)
        p.set_input_bound(DemoIn.THETA, new_low=0.5, new_high=2.5)
        return p
    pass


class DemoNet(nn.Module):
    def __init__(self, dom: AbsDom):
        super().__init__()

        self.input_size = len(DemoIn)
        self.output_size = len(DemoOut)
        self.hidden_sizes = [2]
        self.n_layers = len(self.hidden_sizes) + 1

        self.acti = dom.ReLU()
        self.all_linears = nn.ModuleList()
        in_sizes = [self.input_size] + self.hidden_sizes
        out_sizes = self.hidden_sizes + [self.output_size]
        for in_size, out_size in zip(in_sizes, out_sizes):
            self.all_linears.append(dom.Linear(in_size, out_size, bias=False))  # simplify for motivating example, no bias
        return

    def __str__(self):
        """ Just print everything for information. """
        ss = [
            '--- DemoNet ---',
            'Num layers: %d (i.e. hidden + output, excluding input layer)' % self.n_layers,
            'Input size: %d' % self.input_size,
            'Hidden sizes (len %d): ' % len(self.hidden_sizes) + str(self.hidden_sizes),
            'Output size: %d' % self.output_size,
            'Activation: %s' % self.acti,
            '--- End of AcasNet ---'
        ]
        return '\n'.join(ss)

    def forward(self, x: Union[Tensor, AbsEle]) -> Union[Tensor, AbsEle]:
        """ Normalization and Denomalization are called outside this method. """
        for lin in self.all_linears[:-1]:
            x = lin(x)
            x = self.acti(x)

        x = self.all_linears[-1](x)
        return x
    pass


def demo_net_inited(dom: AbsDom) -> DemoNet:
    net = DemoNet(dom)

    # manually assign the weights
    with torch.no_grad():
        lin1 = net.all_linears[0]
        lin2 = net.all_linears[1]

        lin1.weight.data[0][0] = 1
        lin1.weight.data[0][1] = 0.5
        lin1.weight.data[1][0] = 1
        lin1.weight.data[1][1] = -1

        lin2.weight.data[0][0] = 1
        lin2.weight.data[0][1] = -1
        lin2.weight.data[1][0] = 0.5
        lin2.weight.data[1][1] = 1
    return net


def go(dom: AbsDom, net: DemoNet, prop: DemoProp, _lb: Tensor, _ub: Tensor) -> Tensor:
    ins = dom.Ele.by_intvl(_lb, _ub)
    outs = net(ins)
    return prop.safe_dist(outs)


def demo_analysis():
    print('Analyzing the initial network and safety properties.')
    dom = IntervalDom()

    prop = DemoProp.property12(dom)
    v = Bisecter(dom, prop)
    net = demo_net_inited(dom).to(device)

    in_lbs, in_ubs = prop.lbub(device)
    boxes_lb, boxes_ub = in_lbs, in_ubs

    viols_lb, viols_ub = v.verify(boxes_lb, boxes_ub, None, net)
    print('Found Violation LB:', viols_lb[:10])
    print('Found Violation UB:', viols_ub[:10])

    pts = torch.tensor([[4., 1.]])
    print('Picked violation pts input:', pts)
    print('Picked violation pts output:', net(pts))

    def _inspect_refined(refined_lbs, refined_ubs, title: str):
        print('=====', title, '=====')
        print('Refined LBs:', refined_lbs)
        print('Refined UBs:', refined_ubs)
        dists = go(dom, net, prop, refined_lbs, refined_ubs)
        print(f'Dists: {dists}')
        print(f'Avg dist: {dists.mean()}, Max dist: {dists.max()}')
        print('=====', title, '=====')
        print()
        return

    # (1) inspect right after initialization
    ins = dom.Ele.by_intvl(boxes_lb, boxes_ub)
    outs = net(ins)
    print('Original LB:', boxes_lb)
    print('Original UB:', boxes_ub)
    print('Out LB after initialization:', outs.lb())
    print('Out UB after initialization:', outs.ub())
    _inspect_refined(boxes_lb, boxes_ub, 'Right after initialization')

    # (2) manually specify to split into 2 pieces, along dim 1
    _inspect_refined(torch.tensor([
        [0., 0.5],
        [2.5, 0.5],
    ]), torch.tensor([
        [2.5, 2.5],
        [5., 2.5],
    ]), 'Manual Split into 2, along dim1')  # intvl: max 9.375

    # (3) manually specify to split into 2 pieces, along dim 2
    _inspect_refined(torch.tensor([
        [0., 0.5],
        [0., 1.5],
    ]), torch.tensor([
        [5., 1.5],
        [5., 2.5],
    ]), 'Manual Split into 2, along dim2')  # intvl: max 11.625

    # (4) manually split into 4 pieces, all along dim1
    _inspect_refined(torch.tensor([
        [0., 0.5],
        [1.25, 0.5],
        [2.5, 0.5],
        [3.75, 0.5],
    ]), torch.tensor([
        [1.25, 2.5],
        [2.5, 2.5],
        [3.75, 2.5],
        [5., 2.5],
    ]), 'Manual Split into 4, all along dim1')  # intvl: max 8.125

    # (5) manually split into 4 pieces, all along dim2
    _inspect_refined(torch.tensor([
        [0., 0.5],
        [0., 1.0],
        [0., 1.5],
        [0., 2.0],
    ]), torch.tensor([
        [5., 1.0],
        [5., 1.5],
        [5., 2.0],
        [5., 2.5],
    ]), 'Manual Split into 4, all along dim2')  # intvl: max 11.5

    # (6) manually split into 4 pieces, along both dim1 and dim2
    _inspect_refined(torch.tensor([
        [0., 0.5],
        [0., 0.5],
        [2.5, 1.5],
        [2.5, 1.5],
    ]), torch.tensor([
        [2.5, 1.5],
        [2.5, 1.5],
        [5., 2.5],
        [5., 2.5],
    ]), 'Manual Split into 4, all both dim1 and dim2')  # intvl: max 6.875

    # (7) inspect heuristic splitting into 2 pieces, automatically
    split_lbs, split_ubs = v.split(boxes_lb, boxes_ub, None, net, batch_size=100, stop_on_k_ops=1)
    _inspect_refined(split_lbs, split_ubs, 'One Step Heuristic splitting')

    # (3) inspect naive splitting into K pieces in every dimension  (not used, too similar to ReluVal..)
    results = []
    for k in range(1, 11):
        split_lbs, split_ubs = naive_split(boxes_lb, boxes_ub, k)
        print(f'===== Naive Splitting into {k} pieces along every dimension =====')
        dists = go(dom, net, prop, split_lbs, split_ubs)
        worst_dist = dists.max()
        avg_dist = dists.mean()
        print('After splitting, dists:', dists)
        print('After splitting into', k, 'pieces, worst distance:', worst_dist, 'mean distance:', avg_dist)
        total_loss = dists.mean()
        print('Total loss after initialization:', total_loss)
        print()
        results.append((k, worst_dist.item(), avg_dist.item()))

    print('\n\nAfter everything:')
    _, worst_base, avg_base = results[0]
    for k, worst, avg in results:
        print(f'({k}, {1.0 - worst/worst_base}, {1.0 - avg/avg_base})')
    return


def naive_split(lb: Tensor, ub: Tensor, K: int) -> Tuple[Tensor, Tensor]:
    """ Naively split input regions into a set of total sub-pieces, K pieces in every dimension.
        This will go exponential very quickly.
    :param lb: <Batch x ...>
    :param ub: <Batch x ...>
    """
    if K == 1:
        return lb, ub

    assert lb.size() == ub.size()
    assert lb.dim() == 2 and ub.dim() == 2
    assert len(lb) == len(ub) == 1
    assert K >= 2

    piece_sizes = (ub - lb) / K
    # print(piece_sizes)

    pivots = [lb]
    for k in range(1, K):
        pivots.append(lb + k * piece_sizes)
    pivots.append(ub)

    pivots = [Tensor(v) for v in pivots]
    pivots = torch.cat(pivots, dim=0)
    pivots = torch.Tensor(pivots).t()
    # print('Pivots:', pivots)

    dim_candidates = []
    for i in range(len(pivots)):
        pivots_row = pivots[i]
        candidates = []
        for j in range(1, len(pivots_row)):
            candidates.append((pivots_row[j-1].item(), pivots_row[j].item()))
        dim_candidates.append(candidates)
    # print(dim_candidates)

    splitted = itertools.product(*dim_candidates)
    outs = []
    for s in splitted:
        t = torch.tensor(s)
        outs.append(t)

    lbub = torch.stack(outs, dim=0)
    lbs = lbub[..., 0]
    ubs = lbub[..., 1]
    # print(lbs)
    # print(ubs)
    return lbs, ubs


def art(top_k: int = 10, lr: float = 1e-2, batch_size: int = 16):
    """ VErification Guided Abstracted Learning towards one prop: Train to get all split regions safe.
    :param top_k: if too large, later training epochs will incur many unnecessary refinements
    :param lr: 1e-3 needs 35 epochs, 1e-2 needs 11 epochs.
    :return: <spent epochs, spent time in seconds>
    """
    dom = IntervalDom()
    print(f'Using top_k={top_k}, lr={lr}, batch_size={batch_size}.')

    prop = DemoProp.property12(dom)
    print('For 2-layer ReLU network, it should sat props:', prop.name)
    v = Bisecter(dom, prop)
    net = demo_net_inited(dom).to(device)

    in_lbs, in_ubs = prop.lbub(device)
    boxes_lb, boxes_ub = in_lbs, in_ubs

    opti = Adam(net.parameters(), lr=lr)

    epoch = 0
    total_loss = -1
    start = timer()

    orig_dists = go(dom, net, prop, boxes_lb, boxes_ub)
    print('Before everything, the mean/max dist are', orig_dists.mean(), orig_dists.max())

    results = []
    with torch.no_grad():
        dists = go(dom, net, prop, boxes_lb, boxes_ub)
        results.append((0, dists.max().item()))

    while total_loss != 0.:
        epoch += 1

        trainset = AbsData(boxes_lb, boxes_ub)
        with torch.no_grad():
            dists = go(dom, net, prop, boxes_lb, boxes_ub)
        print(f'[{utils.time_since(start)}] At epoch {epoch}: Loaded {len(trainset)} pieces of boxes for training,',
              f'min loss {dists.min()}, max loss {dists.max()}.')

        trainset_loader = data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
        nbatches = ceil(len(trainset) / batch_size)

        for i, (batch_lb, batch_ub) in enumerate(trainset_loader):
            print(f'\rEpoch {epoch}: {i / nbatches * 100 :.2f}%', end='')
            opti.zero_grad()
            dists = go(dom, net, prop, batch_lb, batch_ub)
            loss = dists.mean()
            loss.backward()
            opti.step()

        # refine inputs
        boxes_lb, boxes_ub = v.split(boxes_lb, boxes_ub, None, net, top_k, stop_on_k_ops=1)

        with torch.no_grad():
            dists = go(dom, net, prop, boxes_lb, boxes_ub)
            total_loss = dists.mean()
            results.append((epoch, dists.max().item()))

        print(f'\r[{utils.time_since(start)}] After epoch {epoch}: total loss {total_loss},'
              f'min dist {dists.min()}, max dist {dists.max()}.')
        pass

    print()
    for e, worst in results:
        print(f'({e}, {worst})')
    return epoch, timer() - start


if __name__ == '__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # demo_analysis()
    art()
    pass
