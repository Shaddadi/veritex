""" An over-approximation soundness 'bug' is observed during collision experiments, where there could exist some outputs
    not covered by the over-approximation via DeepPoly domain with MaxPool layer. The bug is reproduced in this file.

    Bug reproduction:
        Assume a 1d vector [a, b, c] is passed to MaxPool1d with kernel size 2 and stride 1. So the output will be
    a size 2 vector [max(a, b), max(b, c)]. Then it is possible that via abstraction, the same bounds are returned
    for both elements, such that they are cancelled out in later fully-connected layer computations. This issue is
    likely to be due to numerical instability -- when the coefficients are cancelled out, small numerical error may
    make a big difference.
"""

import sys
from pathlib import Path

import torch
from diffabs import DeeppolyDom

sys.path.append(str(Path(__file__).resolve().parent.parent))

from art import collision as c
from art.utils import sample_points

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def test_abstraction_soundness():
    """ Validate that networks and inputs are abstracted correctly using implemented abstract domains.
        However, it turns out that the usage of MaxPool1d makes it rather easy to trigger unsound violations.
        see more details in tests/test_maxpool_soundness.py.
    """
    all_fpaths = list(Path(c.COLLISION_DIR).glob('*.rlv'))
    dom = DeeppolyDom()
    net = c.CollisionMPNet.load(all_fpaths[0], dom, device)  # all nets are the same, just use one

    unstable_cnts = 0
    print('Evaluating abstraction correctness for the saved network:')
    for fpath in all_fpaths:
        prop = c.CollisionProp.load(fpath, dom)
        lb, ub = prop.lbub(device)
        pts = sample_points(lb, ub, 100)
        out_conc = net(pts)

        e = dom.Ele.by_intvl(lb, ub)
        out_lb, out_ub = net(e).gamma()

        threshold = 1e-5  # allow some numerical error
        diff_lb = out_lb - out_conc
        diff_ub = out_conc - out_ub
        # print(diff_lb.max())
        # print(diff_ub.max())
        if diff_lb.max() >= threshold or diff_ub.max() >= threshold:
            unstable_cnts += 1
            print(f'Network {fpath.absolute().name} found unsound cases (due to numerical instability because of MaxPool?)')
            # print(diff_lb.max(), diff_ub.max())
    print(f'-- Eventually, {unstable_cnts} networks seems to be unstable.')
    return


def test_sample_violation():
    """ It suffices to generate such violations by sampling.

        Having both MaxPool1d and FC2 is necessary to reproduce the bug. FC1 must have bias to easily reproduce the
        bug while FC2 may have no bias. Eps = 1e-4 is maximal magnitude to reproduce the bug because the weights and
        input bounds are initialized small.
    """
    dom = DeeppolyDom()
    err_eps = 1e-4

    in_neurons = 1
    fc1_neurons = 3
    kernel_size, stride = 2, 1
    out_neurons = 1

    lb = torch.tensor([[0.1]])
    ub = torch.tensor([[0.12]])

    # fixed
    fc2_neurons = (fc1_neurons - kernel_size) / stride + 1
    assert int(fc2_neurons) == fc2_neurons
    fc2_neurons = int(fc2_neurons)  # if using MaxPool1d
    # fc2_neurons = fc1_neurons  # if not using MaxPool1d

    fc1 = dom.Linear(in_neurons, fc1_neurons, bias=True)
    relu = dom.ReLU()
    mp = dom.MaxPool1d(kernel_size=kernel_size, stride=stride)
    fc2 = dom.Linear(fc2_neurons, out_neurons, bias=False)

    def forward(x):
        x = fc1(x)
        x = relu(x)

        x = x.unsqueeze(dim=1)
        x = mp(x)
        x = x.squeeze(dim=1)

        x = fc2(x)
        return x

    def reset_params():
        fc1.reset_parameters()
        fc2.reset_parameters()
        return

    k = 0
    while True:
        k += 1
        reset_params()

        pts = sample_points(lb, ub, 10000)
        e = dom.Ele.by_intvl(lb, ub)

        out_conc = forward(pts)
        out_lb, out_ub = forward(e).gamma()

        if (out_lb <= out_conc + err_eps).all():
            continue

        print(f'After {k} resets')
        print('LB <= conc?', (out_lb <= out_conc + err_eps).all())
        print('LB <= conc? detail', out_lb <= out_conc + err_eps)

        bits = out_conc + err_eps <= out_lb
        bits = bits.any(dim=1)  # any dimension violation is sufficient
        idxs = bits.nonzero().squeeze(dim=1)

        idx = idxs[0]  # just pick the 1st one to debug
        viol_in = pts[[idx]]
        viol_out = out_conc[[idx]]
        print('conc in:', viol_in.squeeze().item())
        print('out lb:', out_lb.squeeze().item())
        print('out ub:', out_ub.squeeze().item())
        print('conc out:', viol_out.squeeze().item())

        torch.save([fc1, fc2, viol_in], 'error_ctx.pt')
        break
    return


def test_violation_example():
    """ Demonstrate via a handcoded example. """
    dom = DeeppolyDom()
    err_eps = 1e-4

    in_neurons = 1
    fc1_neurons = 3
    kernel_size, stride = 2, 1  # stride must be < kernel size, so as to allow overlapping
    fc2_neurons = (fc1_neurons - kernel_size) / stride + 1  # formula: (W - F + 2P) / S + 1
    assert int(fc2_neurons) == fc2_neurons
    fc2_neurons = int(fc2_neurons)
    out_neurons = 1

    lb = torch.tensor([[0.1]])
    ub = torch.tensor([[0.12]])
    pt = torch.tensor([[0.1010]])

    fc1 = dom.Linear(in_neurons, fc1_neurons)
    fc1.weight.data = torch.tensor([
        [0.9624],
        [-0.6785],
        [0.9087]
    ])
    fc1.bias.data = torch.tensor([0.3255, 0.7965, 0.6321])

    relu = dom.ReLU()
    mp = dom.MaxPool1d(kernel_size=kernel_size, stride=stride)

    fc2 = dom.Linear(fc2_neurons, out_neurons, bias=False)
    fc2.weight.data = torch.tensor([-0.6859, -0.4253])

    def forward(x):
        x = fc1(x)
        x = relu(x)

        x = x.unsqueeze(dim=1)
        x = mp(x)
        x = x.squeeze(dim=1)

        x = fc2(x)
        return x

    print('in pt:', pt.squeeze().item())
    print('in lb:', lb.squeeze().item())
    print('in ub:', ub.squeeze().item())

    e = dom.Ele.by_intvl(lb, ub)
    out = forward(pt)
    out_e = forward(e)
    out_lb, out_ub = out_e.gamma()

    assert not (out_lb <= out + err_eps).all()
    print('out pt:', out.squeeze().item())
    print('out lb:', out_lb.squeeze().item())
    print('out ub:', out_ub.squeeze().item())
    return
