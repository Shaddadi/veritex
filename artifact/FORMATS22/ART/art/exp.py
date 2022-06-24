""" Common utilities and functions used in multiple experiments. """

import argparse
import logging
import random
import signal
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union, List, Optional

import numpy as np
import torch
from torch import Tensor
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils import data

from diffabs import DeeppolyDom, IntervalDom
from diffabs.utils import valid_lb_ub


class PseudoLenDataset(data.Dataset, ABC):
    """ The 'claimed_len' in AbsIns and ConcIns are used to enumerate two datasets simultaneously with true shuffling
        during joint-training.

        torch.data.ConcatDataset cannot do this because it may zip data points from two arbitrary dataset, which would
        result in [Abs, Abs], [Conc, Conc], [Abs, Conc], [Conc, Abs] in the same enumeration. So it is only for two
        homogeneous datasets.

        I was using Combined(Abs, Conc) that re-enumerate one dataset from beginning until both are enumerated. However,
        this is not true shuffling. No matter what shuffled enumeration order it is, idx-1 Abs and idx-1 Conc are always
        zipped together.

        With an extra variable that tracks the claimed length, it can have individual enumeration order for both
        datasets, thus achieving true shuffling.
    """
    def __init__(self, pivot_ls):
        self.pivot_ls = pivot_ls
        self.claimed_len = self.real_len()
        return

    def real_len(self):
        return len(self.pivot_ls)

    def reset_claimed_len(self):
        self.claimed_len = self.real_len()
        return

    def __len__(self):
        """ Allowing enumerating for more than once, so as to co-train with ConcIns. """
        return self.claimed_len

    def __getitem__(self, idx):
        """ There is only split sub-region, no label. """
        return self._getitem(idx % self.real_len())

    @abstractmethod
    def _getitem(self, idx):
        raise NotImplementedError()
    pass


class AbsIns(PseudoLenDataset):
    """ Storing the split LB/UB boxes/abstractions. """
    def __init__(self, boxes_lb: Tensor, boxes_ub: Tensor, boxes_extra: Tensor = None):
        assert valid_lb_ub(boxes_lb, boxes_ub)
        self.boxes_lb = boxes_lb
        self.boxes_ub = boxes_ub
        self.boxes_extra = boxes_extra
        super().__init__(self.boxes_lb)
        return

    def _getitem(self, idx):
        if self.boxes_extra is None:
            return self.boxes_lb[idx], self.boxes_ub[idx]
        else:
            return self.boxes_lb[idx], self.boxes_ub[idx], self.boxes_extra[idx]
    pass


class ConcIns(PseudoLenDataset):
    """ Storing the concrete data points """
    def __init__(self, inputs: Union[List[Tensor], Tensor], labels: Union[List[Tensor], Tensor]):
        assert len(inputs) == len(labels)
        self.inputs = inputs
        self.labels = labels
        super().__init__(self.inputs)
        return

    def _getitem(self, idx):
        return self.inputs[idx], self.labels[idx]
    pass


class ExpArgParser(argparse.ArgumentParser):
    def __init__(self, log_dir: Optional[str], *args, **kwargs):
        """ Override constructor to add more arguments or modify existing defaults.
        :param log_dir: if not None, the directory for log file to dump, the log file name is fixed
        """
        super().__init__(*args, **kwargs)
        self.log_dir = log_dir

        # experiment hyper-parameters
        self.add_argument('--exp_fn', type=str,
                          help='the experiment function to run')
        self.add_argument('--seed', type=int, default=None,
                          help='the random seed for all')

        # art hyper-parameters
        self.add_argument('--dom', type=str, choices=['deeppoly', 'interval'], default='deeppoly',
                          help='the abstract domain to use')
        self.add_argument('--start_abs_cnt', type=int, default=5000,
                          help='do some refinement before training to have more training data')
        self.add_argument('--max_abs_cnt', type=int, default=10000,
                          help='stop refinement after exceeding this many abstractions')
        self.add_argument('--refine_top_k', type=int, default=200,
                          help='select top k abstractions to refine every time')
        self.add_argument('--tiny_width', type=float, default=1e-3,
                          help='refine a dimension only when its width still > this tiny_width')

        # training hyper-parameters
        self.add_argument('--lr', type=float, default=1e-3,
                          help='initial learning rate during training')
        self.add_argument('--batch_size', type=int, default=32,
                          help='mini batch size during each training epoch')
        self.add_argument('--min_epochs', type=int, default=90,
                          help='at least run this many epochs for sufficient training')
        self.add_argument('--max_epochs', type=int, default=100,
                          help='at most run this many epochs before too long')

        # training flags
        self.add_argument('--use_scheduler', action='store_true', default=False,
                          help='using learning rate scheduler during training')
        self.add_argument('--no_pts', action='store_true', default=False,
                          help='not using concrete sampled/prepared points during training')
        self.add_argument('--no_abs', action='store_true', default=False,
                          help='not using abstractions during training')
        self.add_argument('--no_refine', action='store_true', default=False,
                          help='disable refinement during training')

        # printing flags
        group = self.add_mutually_exclusive_group()
        group.add_argument("--quiet", action="store_true", default=False,
                           help='show warning level logs (default: info)')
        group.add_argument("--debug", action="store_true", default=False,
                           help='show debug level logs (default: info)')
        return

    def parse_args(self, args=None, namespace=None):
        res = super().parse_args(args, namespace)
        self.setup_logger(res)  # extra tasks for our experiments
        if res.seed is not None:
            random_seed(res.seed)
        self.setup_rest(res)
        return res

    def setup_logger(self, args: argparse.Namespace):
        logger = logging.getLogger()
        formatter = logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s')

        if args.quiet:
            # default to be warning level
            pass
        elif args.debug:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)

        timestamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        args.stamp = f'{args.exp_fn}-{timestamp}'
        logger.handlers = []  # reset, otherwise it may duplicate many times when calling setup_logger() multiple times
        if self.log_dir is not None:
            log_path = Path(self.log_dir, f'{args.stamp}.log')
            file_handler = logging.FileHandler(log_path)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        return

    @abstractmethod
    def setup_rest(self, args: argparse.Namespace):
        """ Override this method to set up those not easily specified via command line arguments. """
        # validation
        assert not (args.no_pts and args.no_abs), 'training what?'

        args.dom = {
            'deeppoly': DeeppolyDom(),
            'interval': IntervalDom()
        }[args.dom]

        if args.use_scheduler:
            # having a scheduler does improve the accuracy quite a bit
            args.scheduler_fn = lambda opti: ReduceLROnPlateau(opti, factor=0.8, patience=10)
        else:
            args.scheduler_fn = lambda opti: None
        return
    pass


def random_seed(seed):
    """ Set random seed for all. """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    return


class timeout:
    """ Raise error when timeout. Following that in <https://stackoverflow.com/a/22348885>.
    Usage:
        try:
            with timeout(sec=1):
                ...
        except TimeoutError:
            ...
    """
    def __init__(self, sec):
        self.seconds = sec
        self.error_message = 'Timeout after %d seconds' % sec
        return

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)
        return

    def __exit__(self, type, value, traceback):
        signal.alarm(0)
        return
    pass
