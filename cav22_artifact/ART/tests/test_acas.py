import random
import sys
import tempfile
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from diffabs import AbsDom, DeeppolyDom, IntervalDom

sys.path.append(str(Path(__file__).resolve().parent.parent))

from art.acas import AcasProp, AcasNet


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def test_load_and_save(ntimes: int = 10):
    dom = DeeppolyDom()
    prop = AcasProp.property2(dom)
    all_fpaths = prop.applicable_net_paths()

    for _ in range(ntimes):
        fpath = random.choice(all_fpaths)
        # print('Picked nnet file:', fpath)
        net1, mins1, maxs1 = AcasNet.load_nnet(fpath, dom, device)
        # print('Loaded:', net1)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir, 'tmp.nnet')

            net1.save_nnet(tmp_path, mins1, maxs1)
            net2, mins2, maxs2 = AcasNet.load_nnet(tmp_path, dom, device)

            assert mins1 == mins2
            assert maxs1 == maxs2

            assert len(net1.all_linears) == len(net2.all_linears)
            for lin1, lin2 in zip(net1.all_linears, net2.all_linears):
                assert torch.allclose(lin1.weight.data, lin2.weight.data)
                assert torch.allclose(lin1.bias.data, lin2.bias.data)
    return


def _tc1(dom: AbsDom):
    """ Validate that the AcasNet implementation is correct given degenerated interval inputs. """
    net = AcasNet(dom, 2, 1, [2]).to(device)
    for _ in range(10):
        r1 = random.random()
        r2 = random.random()

        inputs = torch.tensor([[r1, r2]], device=device)
        outs = net(inputs)
        outs_lb, outs_ub = net(dom.Ele.by_intvl(inputs, inputs)).gamma()

        assert torch.allclose(outs, outs_lb)
        assert torch.allclose(outs, outs_ub)
    return


def test_net_impl_1():
    dp = DeeppolyDom()
    vi = IntervalDom()

    _tc1(vi)
    _tc1(dp)
    return


def test_net_impl_2():
    """ Validate that my AcasNet implementation is correct given real interval inputs. """
    dom = IntervalDom()

    net = AcasNet(dom, 2, 1, [2]).to(device)
    inputs = torch.tensor([
        [[-2, -1], [-1, 1]],
        [[-0.5, 0.5], [1.5, 3]]
    ], device=device)
    inputs_lb = inputs[:, :, 0]
    inputs_ub = inputs[:, :, 1]

    with torch.no_grad():
        lin0 = net.all_linears[0]
        lin0.weight[0][0] = -0.5
        lin0.weight[0][1] = 0.5
        lin0.bias[0] = -1

        lin0.weight[1][0] = 0.5
        lin0.weight[1][1] = -0.5
        lin0.bias[1] = 1

        lin1 = net.all_linears[1]
        lin1.weight[0][0] = -1
        lin1.weight[0][1] = 1
        lin1.bias[0] = -1

    outs_lb, outs_ub = net(dom.Ele.by_intvl(inputs_lb, inputs_ub)).gamma()
    answer = torch.tensor([
        [[-1.5, 0]],
        [[-1.75, -0.5]]
    ], device=device)
    answer_lb = answer[:, :, 0]
    answer_ub = answer[:, :, 1]

    assert torch.equal(outs_lb, answer_lb)
    assert torch.equal(outs_ub, answer_ub)
    return


def _tc3(dom: AbsDom):
    """ Validate that my AcasNet module can be optimized. """
    inputs = torch.randn(2, 2, 2, device=device)
    inputs_lb, _ = torch.min(inputs, dim=-1)
    inputs_ub, _ = torch.max(inputs, dim=-1)
    ins = dom.Ele.by_intvl(inputs_lb, inputs_ub)

    mse = nn.MSELoss()

    def _loss(outputs_lb):
        lows = outputs_lb[:, 0]
        distances = 0 - lows
        distances = F.relu(distances)
        prop = torch.zeros_like(distances)
        return mse(distances, prop)

    while True:
        net = AcasNet(dom, 2, 2, [2]).to(device)
        with torch.no_grad():
            outputs_lb, outputs_ub = net(ins).gamma()

        if _loss(outputs_lb) > 0:
            break

    # Now the network has something to optimize
    print('===== TC3: =====')
    print('Using inputs LB:', inputs_lb)
    print('Using inputs UB:', inputs_ub)
    print('Before any optimization, the approximated output is:')
    print('Outputs LB:', outputs_lb)
    print('Outputs UB:', outputs_ub)

    opti = torch.optim.Adam(net.parameters(), lr=0.1)
    retrained = 0
    while True:
        opti.zero_grad()
        outputs_lb, outputs_ub = net(ins).gamma()
        loss = _loss(outputs_lb)
        if loss <= 0:
            # until the final output's 1st element is >= 0
            break

        loss.backward()
        opti.step()
        retrained += 1
        print('Iter', retrained, '- loss', loss.item())
        pass

    with torch.no_grad():
        print('All optimized after %d retrains. Now the final outputs 1st element should be >= 0.' % retrained)
        outputs_lb, outputs_ub = net(ins).gamma()
        print('Outputs LB:', outputs_lb)
        print('Outputs UB:', outputs_ub)
        assert (outputs_lb[:, 0] >= 0.).all()
    return retrained


def test_acas_net_optimizable():
    dp = DeeppolyDom()
    vi = IntervalDom()

    _tc3(vi)
    _tc3(dp)
    return


def _tc4(dom: AbsDom):
    """ Validate that my AcasNet module can be optimized at the inputs. """
    mse = nn.MSELoss()
    max_retries = 100
    max_iters = 30  # at each retry, train at most 100 iterations

    def _loss(outputs_lb):
        lows = outputs_lb[..., 0]
        distances = 0 - lows
        distances = F.relu(distances)
        prop = torch.zeros_like(distances)
        return mse(distances, prop)

    retried = 0
    while retried < max_retries:
        # it is possible to get inputs optimized to some local area, thus retry multiple times
        net = AcasNet(dom, 2, 2, [2]).to(device)

        inputs = torch.randn(2, 2, 2, device=device)
        inputs_lb, _ = torch.min(inputs, dim=-1)
        inputs_ub, _ = torch.max(inputs, dim=-1)
        inputs_lb = inputs_lb.requires_grad_()
        inputs_ub = inputs_ub.requires_grad_()
        ins = dom.Ele.by_intvl(inputs_lb, inputs_ub)

        with torch.no_grad():
            outputs_lb, outputs_ub = net(ins).gamma()

        if _loss(outputs_lb) <= 0:
            # found something to optimize
            continue

        retried += 1

        # Now the network has something to optimize
        print(f'\n===== TC4: ({retried}th try) =====')
        print('Using inputs LB:', inputs_lb)
        print('Using inputs UB:', inputs_ub)
        print('Before any optimization, the approximated output is:')
        print('Outputs LB:', outputs_lb)
        print('Outputs UB:', outputs_ub)

        # This sometimes work and sometimes doesn't. It may stuck on a fixed loss and never decrease anymore.
        orig_inputs_lb = inputs_lb.clone()
        orig_inputs_ub = inputs_ub.clone()
        opti = torch.optim.Adam([inputs_lb, inputs_ub], lr=0.1)
        iters = 0
        while iters < max_iters:
            iters += 1

            # after optimization, lb ≤ ub may be violated
            _inputs_lbub = torch.stack((inputs_lb, inputs_ub), dim=-1)
            _inputs_lb, _ = torch.min(_inputs_lbub, dim=-1)
            _inputs_ub, _ = torch.max(_inputs_lbub, dim=-1)
            ins = dom.Ele.by_intvl(_inputs_lb, _inputs_ub)

            opti.zero_grad()
            outputs_lb, outputs_ub = net(ins).gamma()
            loss = _loss(outputs_lb)
            if loss <= 0:
                # until the final output's 1st element is >= 0
                break

            loss.backward()
            opti.step()
            print(f'Iter {iters} - loss {loss.item()}')

        if iters < max_iters:
            # successfully trained
            break

    assert retried < max_retries

    with torch.no_grad():
        print(f'At {retried} retry, all optimized after {iters} iterations. ' +
              f'Now the outputs 1st element should be >= 0 given the latest input.')
        outputs_lb, outputs_ub = net(ins).gamma()
        print('Outputs LB:', outputs_lb)
        print('Outputs UB:', outputs_ub)
        print('Original inputs LB:', orig_inputs_lb)
        print('Optimized inputs LB:', inputs_lb)
        print('Original inputs UB:', orig_inputs_ub)
        print('Optimized inputs UB:', inputs_ub)
        assert (outputs_lb[:, 0] >= 0.).all()
    return


def test_acas_input_optimizable():
    dp = DeeppolyDom()
    vi = IntervalDom()

    _tc4(vi)
    _tc4(dp)
    return
