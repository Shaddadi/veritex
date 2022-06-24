""" Not fully examined in the latest version of code.. """

import os
import sys
from pathlib import Path
from typing import List

import torch
from torch import nn

from diffabs import DeeppolyDom, IntervalDom

sys.path.append(str(Path(__file__).resolve().parent.parent))

from art.acas import AcasNet, AcasProp
from art.external_verifier import _CEX, Reluplex, ReluVal


def _errors(arr1: List, arr2: List) -> float:
    from math import sqrt
    assert len(arr1) == len(arr2)
    err = 0.0
    for (n1, n2) in zip(arr1, arr2):
        err += (n1 - n2) ** 2
    err /= len(arr1)
    return sqrt(err)


def test_reluplex(logs_dir: str = './reluplex_logs/'):
    """ Use property 2 logs from Reluplex for thoroughly examination.
        Need to run Reluplex's script 2 first and prepare the logs in proper location.
    :param logs_dir: directory of logs
    """
    if not Path(logs_dir).is_dir():
        print(f'{logs_dir} is not valid path for all logs.')
        return

    dom = DeeppolyDom()

    def validate_normalize(dnn: AcasNet, cex: _CEX, mins, maxs):
        """ Validate that the normalize() function works the same as NNET's normalizeInput/Output(). """
        ni = torch.tensor([cex.inputs])
        ni = dnn.normalize_inputs(ni, mins, maxs)
        ni = ni[0].detach().numpy()
        target = cex.inputs_normed
        err = _errors(ni, target)

        print('My PyTorch normalizing:', ni)
        print('NNET normalizing:', target)
        print('Error:', err)
        return err

    def validate_dnn(dnn, cex):
        """ Validate that the DNN outputs the same result as NNET does. """
        oi = torch.tensor([cex.inputs])
        oo = dnn(oi)
        oo = oo[0].detach().numpy()
        target = cex.nnet_outputs_normed
        err = _errors(oo, target)

        print('PyTorch :', oo)
        print('NNET C++:', target)
        print('Error:', err)
        return err

    def validate_cex(c, log_path):
        id = log_path[-7:-4]  # e.g. 2_1 for property2_stats_2_1.txt
        net_path = './acas_nets/ACASXU_run2a_%s_batch_2000.nnet' % id
        print(net_path)
        dnn, mins, maxs = AcasNet.load_nnet(net_path, dom)

        err1 = validate_normalize(dnn, c, mins, maxs)
        print('---')
        err2 = validate_dnn(dnn, c)
        print()
        return err1, err2

    reluplex = Reluplex()
    log_files = [fn for fn in os.listdir(logs_dir) if not fn.endswith('_summary.txt')]
    all_cexs = []
    err1s = []
    err2s = []
    for log_name in log_files:
        with open(Path(logs_dir, log_name), 'r') as f:
            log_data = f.read()

        cexs = reluplex.extract(log_data)
        all_cexs.extend(cexs)

        for c in cexs:
            err1, err2 = validate_cex(c, log_name)
            err1s.append(err1)
            err2s.append(err2)
        pass

    print('Errors for normalization:')
    for err in err1s:
        print(err)
    print('Avg:', sum(err1s) / len(err1s))

    print('Errors for forward propagation:')
    for err in err2s:
        print(err)
    print('Avg:', sum(err2s) / len(err2s))

    print('Example:')
    print(all_cexs[0])
    return


def test_reluval(logs_dir: str = './reluval_logs/'):
    """ Use property 2 logs from ReluVal for thoroughly examination.
        Need to run ReluVal's script 2 first and prepare the logs in proper location.
    :param logs_dir: directory of logs
    """
    if not Path(logs_dir).is_dir():
        print(f'{logs_dir} is not valid path for all logs.')
        return

    dom = IntervalDom()

    def validate_dnn(dnn, cex):
        """ Validate that the DNN outputs the same result as NNET does. """
        oi = torch.tensor([cex.inputs])
        oo = dnn(oi)
        oo = oo[0].detach().numpy()
        target = cex.outputs
        err = _errors(oo, target)

        print('My PyTorch:', oo)
        print('ReluVal C++:', target)
        print('Error:', err)
        return err

    def validate_by_prop(dnn, cex, prop_id: int = 2):
        """ It seems the computed outputs are quite different (10^-2 error). So confirm it's true CEX instead? """
        oi = torch.tensor([cex.inputs])
        oo = dnn(oi)

        if prop_id != 2:
            raise NotImplementedError()
        prop = AcasProp.property2(dom)

        e = dom.Ele.by_intvl(oo, oo)
        dist = prop.safe_dist(e)

        mse = nn.MSELoss()
        loss = mse(dist, torch.zeros_like(dist))
        print(f'My PyTorch loss for property{prop_id}: {loss}')
        return loss

    def validate_cex(c, log_path):
        log_name = Path(log_path).name
        prefix = 'ACASXU_run2a_'
        assert prefix in log_name
        id = log_name[len(prefix):len(prefix) + 3]  # e.g. 2_1 for ACASXU_run2a_2_1_batch_2000.nnet.log
        net_path = f'./acas_nets/ACASXU_run2a_{id}_batch_2000.nnet'
        print(net_path)
        dnn, mins, maxs = AcasNet.load_nnet(net_path, dom)

        # err = validate_dnn(dnn, c)
        err = validate_by_prop(dnn, c)
        print()
        return err

    reluval = ReluVal()
    log_files = [fn for fn in os.listdir(logs_dir) if fn.endswith('.nnet.log')]
    errs = []
    for log_name in log_files:
        with open(Path(logs_dir, log_name), 'r') as f:
            log_data = f.read()

        cexs = reluval.extract(log_data)
        for c in cexs:
            print('Validing', c)
            err = validate_cex(c, log_name)
            errs.append(err)
        pass

    print('Losses for forward propagation (should be > 0, so that CEX is genuine):')
    for err in errs:
        print(err)
    print('Avg:', sum(errs) / len(errs))
    return


def test_reluval_cex(nitems: int = 5):
    """ Try to call ReluVal and collect its CEX. Validate that things are working. """
    dom = DeeppolyDom()
    reluval = ReluVal()

    prop = AcasProp.property2(dom)
    lb, ub = prop.lbub()
    for npath in prop.applicable_net_paths()[:nitems]:
        print('Using network from path', npath)
        net, bound_mins, bound_maxs = AcasNet.load_nnet(npath, dom)
        cexs = reluval.verify(lb, ub, net, task_name='2')
        print(cexs)

        # validation
        for i in range(len(cexs)):
            print('------ Validating cex', i)
            cex = cexs[i:i + 1]
            cex = net.normalize_inputs(cex, bound_mins, bound_maxs)
            print('CEX:', cex)
            with torch.no_grad():
                out = net(cex)
            print('Concrete Outs:', out)
            assert out.argmax(dim=-1).item() == 0

            absin = dom.Ele.by_intvl(cex, cex)
            with torch.no_grad():
                absout = net(absin)
            print('Distance:', prop.safe_dist(absout))
            print('------')
    return
