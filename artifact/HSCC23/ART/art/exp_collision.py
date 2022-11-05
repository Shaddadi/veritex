""" Experiments of the collision avoidance/detection dataset from Ehlers  """

import copy
import logging
import math
import sys
from argparse import Namespace
from pathlib import Path
from timeit import default_timer as timer
from typing import Optional, List, Tuple

import torch
from torch import Tensor, nn
from torch.optim import Adam
from torch.utils import data

sys.path.append(str(Path(__file__).resolve().parent.parent))

from art.prop import AndProp
from art.bisecter import Bisecter
from art import exp, utils
from art import collision as c


RES_DIR = Path(__file__).resolve().parent.parent / 'results' / 'collision'
RES_DIR.mkdir(parents=True, exist_ok=True)


class CollisionArgParser(exp.ExpArgParser):
    def __init__(self, log_path: Optional[str], *args, **kwargs):
        super().__init__(log_path, *args, **kwargs)

        self.add_argument('--ffnn', type=str, default='50-128-50',
                          help='specify a FFNN with ReLU activations using hidden layer neurons x-y-z..')
        self.add_argument('--accuracy_loss', type=str, choices=['L1', 'MSE', 'CE'], default='CE',
                          help='canonical loss function for concrete points training')
        self.add_argument('--reset_params', action='store_true', default=False,
                          help='start with random weights or provided trained weights when available')
        self.add_argument('--n_props', type=int, default=100,
                          help='consider how many safety margin central points in training, up to 100')
        self.add_argument('--safe_lambda', type=float, default=1.,
                          help='scale the safety losses')
        self.add_argument('--grad_clip', type=float, default=1.,
                          help='avoid too violent grads')
        self.add_argument('--accu_bar', type=float, default=None,
                          help='acceptable accuracy to pick best safety loss, if None, just pick best accuracy')
        self.add_argument('--certify_timeout', type=int, default=30,
                          help='how many seconds to try certifying the trained network against correctness properties')

        self.set_defaults(exp_fn='test_all', use_scheduler=True)
        return

    def setup_rest(self, args: Namespace):
        super().setup_rest(args)

        def ffnn_fn(in_fs: List[int], out_fs: List[int]) -> nn.Module:
            linears = [args.dom.Linear(in_f, out_f) for in_f, out_f in zip(in_fs, out_fs)]
            relu = args.dom.ReLU()
            layers = []
            for lin in linears:
                layers.append(lin)
                layers.append(relu)
            layers = layers[:-1]  # exclude the last ReLU
            return nn.Sequential(*layers)

        if args.ffnn == '':
            eg_fpath = list(c.COLLISION_DIR.glob('*.rlv'))[0]
            args.net_fn = lambda: c.CollisionMPNet.load(eg_fpath, args.dom)
        else:
            hidden_neurons = [int(s) for s in args.ffnn.split('-')]
            in_features = [c.IN_FEATURES] + hidden_neurons
            out_features = hidden_neurons + [c.OUT_FEATURES]
            args.net_fn = lambda: ffnn_fn(in_features, out_features)
        del args.ffnn

        def ce_loss(outs: Tensor, labels: Tensor):
            softmax = nn.Softmax(dim=1)
            ce = nn.CrossEntropyLoss()
            return ce(softmax(outs), labels)

        args.accuracy_loss = {
            'L1': nn.L1Loss(),
            'MSE': nn.MSELoss(),
            'CE': ce_loss
        }[args.accuracy_loss]
        return
    pass


def eval_test(net: nn.Module, testset: c.CollisionData, categories=None) -> float:
    """ Evaluate accuracy on test set. """
    with torch.no_grad():
        outs = net(testset.inputs)
        predicted = outs.argmax(dim=1)
        correct = (predicted == testset.labels).sum().item()
        ratio = correct / len(testset.inputs)

        # per category
        if categories is not None:
            for cat in categories:
                idxs = testset.labels == cat
                cat_predicted = predicted[idxs]
                cat_labels = testset.labels[idxs]
                cat_correct = (cat_predicted == cat_labels).sum().item()
                cat_ratio = math.nan if len(cat_labels) == 0 else cat_correct / len(cat_labels)
                logging.debug(f'--For category {cat}, out of {len(cat_labels)} items, ratio {cat_ratio}')
    return ratio


def train_collision(net: nn.Module, full_props: List[c.CollisionProp], args: Namespace) -> Tuple[int, float, int, float]:
    """ The almost completed skeleton of training Collision Avoidance/Detection networks using ART.
    :return: trained_epochs, train_time, certified, final accuracies
    """
    logging.info(net)
    if args.reset_params:
        try:
            net.reset_params()
        except AttributeError:
            ''' This is possible when creating FFNN on the fly which doesn't have reset_params().
                It's fine since such FFNN is using newly initialized weights.
            '''
            pass

    props_dict = c.cluster_props(full_props)
    large_props = [ps[0] for ps in props_dict.values()]  # pick the largest one for each safety margin base point
    large_props = AndProp(large_props[:args.n_props])
    logging.info(f'Using {len(large_props.props)} largest properties.')

    v = Bisecter(args.dom, large_props)

    def run_abs(batch_abs_lb: Tensor, batch_abs_ub: Tensor, batch_abs_bitmap: Tensor) -> Tensor:
        """ Return the safety distances over abstract domain. """
        batch_abs_ins = args.dom.Ele.by_intvl(batch_abs_lb, batch_abs_ub)
        batch_abs_outs = net(batch_abs_ins)
        return large_props.safe_dist(batch_abs_outs, batch_abs_bitmap)

    in_lb, in_ub = large_props.lbub(device)
    in_bitmap = large_props.bitmap(device)

    # already moved to GPU if necessary
    trainset = c.CollisionData.load(device)
    testset = trainset  # there is only training set, following that in Ehlers 2017

    start = timer()

    if args.no_abs or args.no_refine:
        curr_abs_lb, curr_abs_ub, curr_abs_bitmap = in_lb, in_ub, in_bitmap
    else:
        # refine it at the very beginning to save some steps in later epochs
        curr_abs_lb, curr_abs_ub, curr_abs_bitmap = v.split(in_lb, in_ub, in_bitmap, net, args.refine_top_k,
                                                            # tiny_width=args.tiny_width,
                                                            stop_on_k_all=args.start_abs_cnt)

    opti = Adam(net.parameters(), lr=args.lr)
    scheduler = args.scheduler_fn(opti)  # could be None

    accuracies = []  # epoch 0: ratio
    best_metric = 1e9 if args.accu_bar else -1.
    best_params = None
    certified = False
    epoch = 0
    while True:
        # first, evaluate current model
        logging.info(f'[{utils.time_since(start)}] After epoch {epoch}:')
        if not args.no_pts:
            logging.info(f'Loaded {trainset.real_len()} points for training.')
        if not args.no_abs:
            logging.info(f'Loaded {len(curr_abs_lb)} abstractions for training.')
            with torch.no_grad():
                full_dists = run_abs(curr_abs_lb, curr_abs_ub, curr_abs_bitmap)
            worst_loss = full_dists.max()
            logging.info(f'min loss {full_dists.min()}, max loss {worst_loss}.')
            if worst_loss <= 0.:
                certified = True
                logging.info(f'All {len(curr_abs_lb)} abstractions certified.')
            else:
                _, worst_idx = full_dists.max(dim=0)
                logging.info(f'Max loss at LB: {curr_abs_lb[worst_idx]}, UB: {curr_abs_ub[worst_idx]}.')
                worst_props = large_props.props_of(curr_abs_bitmap[worst_idx])
                logging.info(f'Max loss labels: {[p.larger_category for p in worst_props]}')

        accu = eval_test(net, testset)
        accuracies.append(accu)
        logging.info(f'Test set accuracy {accu}.')
        if args.accu_bar is None or args.no_abs:
            # pick the best accuracy model
            if accu > best_metric:
                best_metric = accu
                best_params = copy.deepcopy(net.state_dict())
        else:
            if accu > args.accu_bar and worst_loss < best_metric:
                best_metric = worst_loss
                best_params = copy.deepcopy(net.state_dict())

        # check termination
        if certified and epoch >= args.min_epochs:
            # all safe and sufficiently trained
            break

        if epoch >= args.max_epochs:
            break

        epoch += 1
        certified = False

        # writting like this because ReduceLROnPlateau do not have get_lr()
        _param_lrs = [group['lr'] for group in opti.param_groups]
        curr_lr = sum(_param_lrs) / len(_param_lrs)
        logging.info(f'\n[{utils.time_since(start)}] Starting epoch {epoch} with lr = {curr_lr}:')

        absset = exp.AbsIns(curr_abs_lb, curr_abs_ub, curr_abs_bitmap)

        # dataset may have expanded, need to update claimed length to date
        if not args.no_pts:
            trainset.reset_claimed_len()
        if not args.no_abs:
            absset.reset_claimed_len()
        if (not args.no_pts) and (not args.no_abs):
            ''' Might simplify this to just using the amount of abstractions, is it unnecessarily complicated? '''
            # need to enumerate both
            max_claimed_len = min(trainset.claimed_len, absset.claimed_len)
            # max_claimed_len = trainset.claimed_len
            trainset.claimed_len = max_claimed_len
            absset.claimed_len = max_claimed_len

        if not args.no_pts:
            # using drop_last may increase accuracy a bit, but decrease safety a bit?
            conc_loader = data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, drop_last=True)
            nbatches = len(conc_loader)
            conc_loader = iter(conc_loader)
        if not args.no_abs:
            # using drop_last may increase accuracy a bit, but decrease safety a bit?
            abs_loader = data.DataLoader(absset, batch_size=args.batch_size, shuffle=True, drop_last=True)
            nbatches = len(abs_loader)  # doesn't matter rewriting len(conc_loader), they are the same
            abs_loader = iter(abs_loader)

        accu_total_loss = 0.
        safe_total_loss = 0.
        for i in range(nbatches):
            opti.zero_grad()
            batch_loss = 0.
            if not args.no_pts:
                batch_inputs, batch_labels = next(conc_loader)
                batch_outputs = net(batch_inputs)
                batch_loss += args.accuracy_loss(batch_outputs, batch_labels)
                accu_total_loss += batch_loss.item()
            if not args.no_abs:
                batch_abs_lb, batch_abs_ub, batch_abs_bitmap = next(abs_loader)
                batch_dists = run_abs(batch_abs_lb, batch_abs_ub, batch_abs_bitmap)
                safe_loss = batch_dists.mean()  # L1, need to upgrade to batch_worsts to unlock loss other than L1
                safe_loss *= args.safe_lambda
                safe_total_loss += safe_loss.item()
                batch_loss += safe_loss
            logging.debug(f'Epoch {epoch}: {i / nbatches * 100 :.2f}%. Batch loss {batch_loss.item()}')
            batch_loss.backward()

            nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)  # doesn't seem to make a difference here..
            opti.step()

        # inspect the trained weights after another epoch
        # meta.inspect_params(net.state_dict())

        accu_total_loss /= nbatches
        safe_total_loss /= nbatches
        if scheduler is not None:
            scheduler.step(accu_total_loss + safe_total_loss)
        logging.info(f'[{utils.time_since(start)}] At epoch {epoch}: avg accuracy training loss {accu_total_loss}, ' +
                     f'safe training loss {safe_total_loss}.')

        # Refine abstractions, note that restart from scratch may output much fewer abstractions thus imprecise.
        if (not args.no_refine) and len(curr_abs_lb) < args.max_abs_cnt:
            curr_abs_lb, curr_abs_ub, curr_abs_bitmap = v.split(curr_abs_lb, curr_abs_ub, curr_abs_bitmap, net,
                                                                args.refine_top_k,
                                                                # tiny_width=args.tiny_width,
                                                                stop_on_k_new=args.refine_top_k)
        pass

    # summarize
    train_time = timer() - start

    if certified and args.n_props == 100:
        # the latest one is certified, use that
        final_accu = accuracies[-1]
        tot_certified = 500
    else:
        # not yet having a certified model, thus pick the one with best accuracy so far and try certify it on all props
        if best_params is not None:
            logging.info(f'Post certify using best metric {best_metric}')
            net.load_state_dict(best_params)

        final_accu = eval_test(net, testset)
        tot_certified = 0
        for i, (k, ps) in enumerate(props_dict.items()):
            assert len(ps) == 5
            for j, p in enumerate(ps):
                tmp_v = Bisecter(args.dom, p)
                in_lb, in_ub = p.lbub(device)
                if tmp_v.try_certify(in_lb, in_ub, None, net, args.batch_size, timeout_sec=args.certify_timeout):
                    tot_certified += (5 - j)
                    logging.info(f'Certified prop based at {k} using {j}th eps, now {tot_certified}/{5*(i+1)}.')
                    break
        pass

    serial_net = nn.Sequential(*[layer.export() for layer in net])  # save exported network in serialization
    torch.save(serial_net.cpu(), Path(RES_DIR, f'trained-{tot_certified}-{final_accu:.4f}-model.pt'))

    accuracies = [f'{v:.4f}' for v in accuracies]
    logging.info(f'Accuracy at every epoch: {accuracies}')
    logging.info(f'After {epoch} epochs / {utils.pp_time(train_time)}, ' +
                 f'eventually the trained network got certified at {tot_certified} / 500 props, ' +
                 f'with {final_accu:.4f} accuracy on test set.')
    return epoch, train_time, tot_certified, final_accu


def test_all(parser: CollisionArgParser):
    """ Q: Show that we can train all previously safe/unsafe networks to safe, and evaluate on the given dataset. """
    defaults = {
        'min_epochs': 20,
    }
    parser.set_defaults(**defaults)
    args = parser.parse_args()
    logging.info(utils.fmt_args(args))

    # since all original network params are the same, we don't need to load many times
    net = args.net_fn().to(device)
    all_props = [c.CollisionProp.load(fpath, args.dom) for fpath in c.COLLISION_DIR.glob('*.rlv')]
    outs = train_collision(net, all_props, args)

    logging.info(f'Final Summary -- Avg <epochs, train_time, certified, accuracy>: {outs}')
    return


if __name__ == '__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    test_defaults = {
        'exp_fn': 'test_all',
        # 'no_refine': True,
    }
    parser = CollisionArgParser(RES_DIR, description='Collision Avoidance Dataset Experiments')
    parser.set_defaults(**test_defaults)
    args, _ = parser.parse_known_args()

    exp_fn = locals()[args.exp_fn]
    start = timer()
    exp_fn(parser)

    logging.info(f'Total Cost Time: {timer() - start}s.\n\n\n')
    pass
