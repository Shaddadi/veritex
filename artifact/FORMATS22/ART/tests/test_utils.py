import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))

from art.utils import sample_points, gen_rnd_points


def test_sample_points(nrow=10, ncol=10, K=1000):
    """ Validate that sampled points are within range. """
    t1t2 = torch.stack((torch.randn(nrow, ncol), torch.randn(nrow, ncol)), dim=-1)
    lb, _ = torch.min(t1t2, dim=-1)
    ub, _ = torch.max(t1t2, dim=-1)
    outs = sample_points(lb, ub, K)

    assert len(outs) == nrow * K
    for i in range(nrow * K):
        row = i % nrow
        for j in range(ncol):
            assert lb[row][j] <= outs[i][j]
            assert outs[i][j] <= ub[row][j]
    return


def test_rnd_points(K=100):
    lb = torch.tensor([[-10., -10.], [0., 0.]])
    ub = torch.tensor([[0., 0.], [10., 10.]])
    pts, _ = gen_rnd_points(lb, ub, None, K)

    neg_pts, pos_pts = pts[0], pts[1]
    assert (neg_pts <= 0.).all()
    assert (pos_pts >= 0.).all()
    return
