""" Evaluate the verification of NN capability using ACAS dataset. See the report of VNN 2020 for a table of detailed
    results from various existing tools at https://sites.google.com/view/vnn20/vnncomp.
"""

import logging
import sys
from argparse import Namespace
from pathlib import Path
from typing import Optional
from timeit import default_timer as timer

import torch

from diffabs import DeeppolyDom

sys.path.append(str(Path(__file__).resolve().parent.parent))

from art import acas, exp
from art.prop import AndProp
from art.bisecter import Bisecter
from art.cluster import Cluster
from art.vnn import VNN20Info
from art.utils import pp_time, fmt_args


RES_DIR = Path(__file__).resolve().parent.parent / 'results' / 'vnn'
RES_DIR.mkdir(parents=True, exist_ok=True)


class VerifyArgParser(exp.ExpArgParser):
    def __init__(self, log_path: Optional[str], *args, **kwargs):
        super().__init__(log_path, *args, **kwargs)

        # verify a specific task
        self.add_argument('--prop', type=int, default=1,
                          help='prop id, from 1 to 10')
        self.add_argument('--net', type=str, default='1-1',
                          help='network id, in the shape of x-y')
        self.add_argument('--timeout_sec', type=int, default=300,
                          help='allowed time for a verifier query')

        # verify a series of tasks
        self.add_argument('--task_i', type=int, default=None,
                          help='first index of the tasks to verify, 0 based, inclusive')
        self.add_argument('--task_j', type=int, default=None,
                          help='last index of the tasks to verify, 0 based, not-inclusive')
        self.add_argument('--series', type=str, choices=['all', 'hard'], default='all',
                          help='verify tasks in the all- or hard- series')

        self.add_argument('--use_new', action='store_true', default=False,
                          help='use new refinement')

        # experiment on first two hard tasks show that 8192 is fastest, 4096 and 16384 are slower
        self.set_defaults(batch_size=8192)
        return
    pass


def _verify(nid: acas.AcasNetID, all_props: AndProp, args: Namespace):
    fpath = nid.fpath()
    net, bound_mins, bound_maxs = acas.AcasNet.load_nnet(fpath, dom, device)
    # logging.info(net)  # no need to print acas network here, all the same

    v = Cluster(dom, all_props) if args.use_new else Bisecter(dom, all_props)

    in_lb, in_ub = all_props.lbub(device)
    in_bitmap = all_props.bitmap(device)
    in_lb = net.normalize_inputs(in_lb, bound_mins, bound_maxs)
    in_ub = net.normalize_inputs(in_ub, bound_mins, bound_maxs)

    res = v.verify(in_lb, in_ub, in_bitmap, net, batch_size=args.batch_size)
    return res


def verify_net_prop(parser: VerifyArgParser):
    """ Verify a specific network w.r.t. a single property. """
    args = parser.parse_args()
    logging.info(fmt_args(args))

    # parse network id
    nums = [int(v) for v in args.net.strip().split('-')]
    nid = acas.AcasNetID(*nums)

    # parse prop id
    if args.prop == 6:
        # 6a and 6b
        all_props = AndProp([acas.AcasProp.property6a(dom), acas.AcasProp.property6b(dom)])
    else:
        prop_method = f'property{args.prop}'
        prop = getattr(acas.AcasProp, prop_method)(dom)
        all_props = AndProp([prop])

    logging.info(f'===== Processing {nid}, verifying one property {all_props.name} =====')
    res = _verify(nid, all_props, args)
    t0 = timer()
    logging.info(f'After {pp_time(timer() - t0)}, verify result -- CEX: {res}\n\n')
    return res


def verify_tasks(parser: VerifyArgParser):
    """ Verify a range of tasks in the VNN-COMP2020 table. """
    args = parser.parse_args()
    logging.info(fmt_args(args))
    assert args.task_i is not None and args.task_j is not None

    info = VNN20Info()
    d = info.results_all if args.series == 'all' else info.results_hard

    low, high = args.task_i, min(len(d.props), args.task_j)
    logging.info(f'Enumerating verification task [{low}, {high}).')
    for i in range(low, high):
        # parse network id
        nid_nums = [int(v) for v in d.nets[i].strip().split('-')]
        nid = acas.AcasNetID(*nid_nums)

        # parse prop id
        prop = int(d.props[i])
        if prop == 6:
            # 6a and 6b
            all_props = AndProp([acas.AcasProp.property6a(dom), acas.AcasProp.property6b(dom)])
        else:
            prop_method = f'property{prop}'
            prop = getattr(acas.AcasProp, prop_method)(dom)
            all_props = AndProp([prop])

        logging.info(f'===== Processing {nid}, verifying one property {all_props.name} =====')
        t0 = timer()
        res = _verify(nid, all_props, args)
        logging.info(f'After {pp_time(timer() - t0)}, verify result -- CEX: {res}\n\n')
    return


def verify_net(parser: VerifyArgParser):
    """ Verify all properties a network should hold at the same time. """
    args = parser.parse_args()
    logging.info(fmt_args(args))

    # parse network id
    nums = [int(v) for v in args.net.strip().split('-')]
    nid = acas.AcasNetID(*nums)

    # should hold for all props
    all_props = AndProp(nid.applicable_props(dom))

    logging.info(f'===== Processing {nid}, verifying all its props {all_props.name} =====')
    t0 = timer()
    res = _verify(nid, all_props, args)
    logging.info(f'After {pp_time(timer() - t0)}, verify result -- CEX: {res}\n\n')
    return res


if __name__ == '__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    dom = DeeppolyDom()

    test_defaults = {
        # 'exp_fn': 'verify_net_prop',
        # 'exp_fn': 'verify_net',
        'exp_fn': 'verify_tasks',

        'task_i': 0,
        'task_j': 2,  # [0, 5) tasks
        'series': 'hard',
    }
    parser = VerifyArgParser(RES_DIR, description='NN Verification Experiment')
    parser.set_defaults(**test_defaults)
    args, _ = parser.parse_known_args()

    exp_fn = locals()[args.exp_fn]
    start = timer()
    exp_fn(parser)

    logging.info(f'Total Cost Time: {timer() - start}s.\n\n\n')
    pass
