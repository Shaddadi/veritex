from argparse import Namespace
from math import floor
from timeit import default_timer as timer
from typing import Optional, Tuple

import torch
from torch import Tensor, cuda

from diffabs.utils import valid_lb_ub


def sample_points(lb: Tensor, ub: Tensor, K: int) -> Tensor:
    """ Uniformly sample K points for each region. Resulting in large batch of states.
    :param lb: Lower bounds, batched
    :param ub: Upper bounds, batched
    :param K: how many pieces to sample
    :return: (Batch * K) x State
    """
    assert valid_lb_ub(lb, ub)
    assert K >= 1

    repeat_dims = [1] * (len(lb.size()) - 1)
    base = lb.repeat(K, *repeat_dims)  # repeat K times in the batch, preserving the rest dimensions
    width = (ub - lb).repeat(K, *repeat_dims)

    coefs = torch.rand_like(base)
    pts = base + coefs * width
    return pts


def gen_rnd_points(lb: Tensor, ub: Tensor, extra: Optional[Tensor], K: int) -> Tuple[Tensor, Optional[Tensor]]:
    """ Different from old sample_points(), the output here maintains the structure. Also accepts extra.
    :param lb: Lower bounds, batched
    :param ub: Upper bounds, batched
    :param extra: e.g., bitmaps for properties in AndProp
    :param K: how many states to per abstraction
    :return: Batch x K x State, together with expanded extra
    """
    assert valid_lb_ub(lb, ub)
    assert K >= 1

    new_size = list(lb.size())
    new_size.insert(1, K)  # Batch x States => Batch x K x States

    base = lb.unsqueeze(dim=1).expand(*new_size)
    width = (ub - lb).unsqueeze(dim=1).expand(*new_size)

    coefs = torch.rand_like(base)
    pts = base + coefs * width

    if extra is None:
        new_extra = None
    else:
        new_size = list(extra.size())
        new_size.insert(1, K)
        new_extra = extra.unsqueeze(dim=1).expand(*new_size)
    return pts, new_extra


def gen_vtx_points(base_lb: Tensor, base_ub: Tensor, extra: Optional[Tensor]) -> Tuple[Tensor, Optional[Tensor]]:
    """ Generate the vertices of a hyper-rectangle bounded by LB/UB.
    :param base_lb/base_ub: batched
    :return: Batch x N x State, where N is the number of vertices in each abstraction
    """
    # TODO a faster way might be using torch.where() to select from LB/UB points with [0, 2^n-1] indices
    all_vtxs = []
    for lb, ub in zip(base_lb, base_ub):
        # basically, a cartesian product of LB/UB on each dimension
        lbub = torch.stack((lb, ub), dim=-1)  # Dim x 2
        vtxs = torch.cartesian_prod(*list(lbub))
        all_vtxs.append(vtxs)
    all_vtxs = torch.stack(all_vtxs, dim=0)

    if extra is None:
        new_extra = None
    else:
        new_size = list(extra.size())
        new_size.insert(1, all_vtxs.shape[1])
        new_extra = extra.unsqueeze(dim=1).expand(*new_size)
    return all_vtxs, new_extra


def total_area(lb: Tensor, ub: Tensor, eps: float = 1e-8, by_batch: bool = False) -> float:
    """ Return the total area constrained by LB/UB. Area = \Sum_{batch}{ \Prod{Element} }.
    :param lb: <Batch x ...>
    :param ub: <Batch x ...>
    :param by_batch: if True, return the areas of individual abstractions
    """
    assert valid_lb_ub(lb, ub)
    diff = ub - lb
    diff += eps  # some dimensions may be degenerated, then * 0 becomes 0.

    while diff.dim() > 1:
        diff = diff.prod(dim=-1)

    if by_batch:
        return diff
    else:
        return diff.sum().item()


def fmt_args(args: Namespace) -> str:
    title = args.stamp
    s = [f'\n===== {title} configuration =====']
    d = vars(args)
    for k, v in d.items():
        if k == 'stamp':
            continue
        s.append(f'  {k}: {v}')
    s.append(f'===== end of {title} configuration =====\n')
    return '\n'.join(s)


def pp_time(duration: float) -> str:
    """
    :param duration: in seconds
    """
    m = floor(duration / 60)
    s = duration - m * 60
    return '%dm %ds (%.3f seconds)' % (m, s, duration)


def time_since(since, existing=None):
    t = timer() - since
    if existing is not None:
        t += existing
    return pp_time(t)


def pp_cuda_mem(stamp: str = '') -> str:
    def sizeof_fmt(num, suffix='B'):
        for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
            if abs(num) < 1024.0:
                return "%3.1f%s%s" % (num, unit, suffix)
            num /= 1024.0
        return "%.1f%s%s" % (num, 'Yi', suffix)

    if not cuda.is_available():
        return ''

    return '\n'.join([
        f'----- {stamp} -----',
        f'Allocated: {sizeof_fmt(cuda.memory_allocated())}',
        f'Max Allocated: {sizeof_fmt(cuda.max_memory_allocated())}',
        f'Cached: {sizeof_fmt(cuda.memory_cached())}',
        f'Max Cached: {sizeof_fmt(cuda.max_memory_cached())}',
        f'----- End of {stamp} -----'
    ])
