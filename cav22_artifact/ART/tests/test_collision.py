import sys
from collections import defaultdict
from pathlib import Path

import torch
from torch import Tensor

from diffabs import DeeppolyDom

sys.path.append(str(Path(__file__).resolve().parent.parent))

from art import collision as c
from art.utils import valid_lb_ub

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def _locate(all_pts: Tensor, pt: Tensor) -> int:
    for i, dp in enumerate(all_pts):
        if torch.allclose(dp, pt):
            return i
    return -1


def test_prop_load():
    """ Validate the loading of correctness properties such that it matches the training set labels. """
    ds = c.CollisionData.load(device)
    all_fpaths = list(Path(c.COLLISION_DIR).glob('*.rlv'))

    cnts = defaultdict(list)
    print('Evaluating property loading (supposed to be robustness props around dataset samples):')
    for fpath in all_fpaths:
        prop = c.CollisionProp.load(fpath, None)

        mid = (prop.bound_mins + prop.bound_maxs) / 2.
        eps = (prop.bound_maxs - prop.bound_mins) / 2.
        mid, eps = mid.to(device), eps.to(device)
        idx = _locate(ds.inputs, mid)
        label = ds.labels[idx]
        cnts[idx].append(eps)
        print(f'-- For {fpath.absolute().name}, prop requires {prop.larger_category}, ' +
              f'mid point located at dataset idx {idx} of label = {label}.')

        assert prop.larger_category == label

    cnts = sorted(cnts.items(), key=lambda p: p[0])
    for k, v in cnts:
        assert len(v) == 5
        print(f'idx {k} -- {len(v)} times -- {v}')
    return


def test_no_prop_conflict():
    """ Is there any safety margin that overlap and with different labels? Then some margin must be violated. """
    all_fpaths = list(Path(c.COLLISION_DIR).glob('*.rlv'))
    all_props = [c.CollisionProp.load(p, None) for p in all_fpaths]
    props_dict = c.cluster_props(all_props)
    all_props = [ps[0] for ps in props_dict.values()]  # pick the first largest prop

    for i, p1 in enumerate(all_props):
        for j in range(i + 1, len(all_props)):
            p2 = all_props[j]

            intersect_mins = torch.max(torch.stack((p1.bound_mins, p2.bound_mins), dim=-1), dim=-1)[0]
            intersect_maxs = torch.min(torch.stack((p1.bound_maxs, p2.bound_maxs), dim=-1), dim=-1)[0]
            if valid_lb_ub(intersect_mins, intersect_maxs):
                # overlapping
                print(f'overlapped, {i} vs {j}')
                assert p1.larger_category == p2.larger_category

    print(f'All validated, no conflicting overlapped props.')
    return


def test_net_load():
    """ Validate the loading of network such that it works on all 3k dataset samples. """
    ds = c.CollisionData.load(device)
    all_fpaths = list(Path(c.COLLISION_DIR).glob('*.rlv'))

    print('Evaluating accuracies for each saved network (supposed to be the same):')
    dom = DeeppolyDom()  # which dom doesn't matter
    for fpath in all_fpaths:
        net = c.CollisionMPNet.load(fpath, dom, device)
        outs = net(ds.inputs)
        predicted = outs.argmax(dim=-1)

        accuracy = len((predicted == ds.labels).nonzero(as_tuple=False)) / len(ds.labels)
        print(f'-- Accuracy for {fpath.absolute().name}: {accuracy}')
        assert accuracy >= 0.99

    # inspect unsafe ones from last run, since all networks are the same, it doesn't matter
    failed_bits = (predicted != ds.labels).nonzero(as_tuple=True)
    failed_ins = ds.inputs[failed_bits]
    failed_labels = ds.labels[failed_bits]
    failed_outs = outs[failed_bits]
    print(f'Failing on following inputs: {failed_ins}')
    print(f'should be label: {failed_labels}')
    print(f'with outpus: {failed_outs}')
    return
