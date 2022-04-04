
import sys
from pathlib import Path

import torch

from diffabs import DeeppolyDom

sys.path.append(str(Path(__file__).resolve().parent.parent))

from art import acas
from art.bisecter import Bisecter


def inspect():
    """ Inspect the saved abstractions for refinement choice, saved during bisecter.verify() after 100 iters.
        It shows that many refinement choices when safe-dist is small are not optimal.
    """
    nid = acas.AcasNetID(3, 3)
    prop = acas.AcasProp.property2(dom)

    fpath = nid.fpath()
    net, _, _ = acas.AcasNet.load_nnet(fpath, dom, device)

    v = Bisecter(dom, prop)

    file_dir = Path(__file__).resolve().parent
    batch_lb, batch_ub, batch_extra, batch_grad = torch.load(file_dir / 'prop2-net3-3-iter100-abs.pyt')

    first_n1, first_n2 = 0, 50  # FIXME
    batch_lb = batch_lb[first_n1:first_n2]
    batch_ub = batch_ub[first_n1:first_n2]
    batch_extra = batch_extra[first_n1:first_n2]
    batch_grad = batch_grad[first_n1:first_n2]

    # explicitly find best bisection dimension
    for i in range(len(batch_lb)):
        one_lb, one_ub, one_extra, one_grad = batch_lb[i], batch_ub[i], batch_extra[i], batch_grad[i]

        orig_e = dom.Ele.by_intvl(one_lb.unsqueeze(dim=0), one_ub.unsqueeze(dim=0))
        orig_dist = prop.safe_dist(net(orig_e))
        assert (orig_dist > 0.).all()

        print(f'For {i}th abstraction: LB {one_lb}, UB {one_ub}, grad {one_grad},')
        print(f'safe dist {orig_dist}')

        dims = len(one_lb)
        for j in range(dims):
            lhs_lb = one_lb.clone()
            lhs_ub = one_ub.clone()
            lhs_ub[j] = (one_lb[j] + one_ub[j]) / 2.

            rhs_lb = one_lb.clone()
            rhs_ub = one_ub.clone()
            rhs_lb[j] = (one_lb[j] + one_ub[j]) / 2.

            refined_lb = torch.stack([lhs_lb, rhs_lb], dim=0)
            refined_ub = torch.stack([lhs_ub, rhs_ub], dim=0)

            refined_e = dom.Ele.by_intvl(refined_lb, refined_ub)
            refined_outs = net(refined_e)
            refined_dists = prop.safe_dist(refined_outs)
            print(f'-- If splitting dim {j}, dists: {refined_dists}')

        width = one_ub - one_lb
        smears = one_grad * width / 2
        _, split_idxs = smears.max(dim=-1)
        print(f'Smear decided dim {split_idxs}')

    # refined = v.by_smear(batch_lb, batch_ub, batch_extra, batch_grad)
    return


if __name__ == '__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    dom = DeeppolyDom()

    inspect()
    pass
