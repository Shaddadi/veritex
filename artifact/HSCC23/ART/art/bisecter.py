""" Apply input bisection for Neural Network verification / certification / falsification. """

import logging
import sys
from pathlib import Path
from typing import Tuple, Optional, Union
from timeit import default_timer as timer

import torch
from torch import Tensor, nn, autograd
from torch.utils.data import DataLoader

from diffabs import AbsDom, AbsData
from diffabs.utils import valid_lb_ub

sys.path.append(str(Path(__file__).resolve().parent.parent))

from art.prop import AbsProp
from art.exp import timeout
from art.utils import total_area, pp_time, sample_points, pp_cuda_mem


def empty_like(t: Tensor) -> Tensor:
    """ Empty tensor (different from torch.empty() for concatenation. """
    return torch.tensor([], device=t.device)


def cat0(*ts: Tensor) -> Optional[Tensor]:
    """ Usage: simplify `torch.cat((ts1, ts2), dim=0)` to `cat0(ts1, ts2)`. """
    if ts[0] is None:
        return None
    return torch.cat(ts, dim=0)


class Bisecter(object):
    """ NN Verifier using bisection based input refinement and over-approximation.

        Basic algorithm:
        (1) For each input region, it tries to prove either safety or violation by over-approximation.
        (2) If failed, the input region is bisected to a total set of smaller ones, via different heuristics.
        (3) All these input regions are maintained in a work list, algorithm ends when the work list is empty.

        In v1, there was also viol_dists that tries to find a region all certified to violate. This turns out worse
        than sampling to check violation. Because as long as it can certify to violate, sampling must find a cex,
        either. After all, if we care about CEX, we need just one of them.
    """

    def __init__(self, domain: AbsDom, prop: AbsProp):
        """
        :param domain: the abstract domain module to use
        """
        self.d = domain
        self.prop = prop
        return

    def _grad_dists_of_batch(self, new_lb: Tensor, new_ub: Tensor, new_extra: Optional[Tensor],
                             forward_fn: nn.Module) -> Tuple[Tensor, Tensor]:
        """ Get the gradient value for each abstraction as heuristic, as long as safe distances. """
        with torch.enable_grad():
            new_lb = new_lb.detach().requires_grad_()
            new_ub = new_ub.detach().requires_grad_()
            if new_lb.grad is not None:
                new_lb.grad.zero_()  # in case any previous grads are unexpectedly accumulated
            if new_ub.grad is not None:
                new_ub.grad.zero_()

            ins = self.d.Ele.by_intvl(new_lb, new_ub)
            outs = forward_fn(ins)

            new_safe_dist = self.prop.safe_dist(outs) if new_extra is None else self.prop.safe_dist(outs, new_extra)
            grad_dist = new_safe_dist

            ''' Sum safe/viol_dists to get one single value for backprop. Otherwise it needs to pass in 'grad_outputs'
                argument for autograd.grad(). e.g., use ones_like(dists).
                1st order summation will distribute the original output distance to each corresponding input.
                After all, it only needs to relatively compare input grads.
            '''
            assert grad_dist.dim() == 1, 'Do I need to squeeze the losses into <Batch> vector first?'
            losses = grad_dist.sum()

            # back-propagate safety loss to inputs
            grads = autograd.grad(losses, [new_lb, new_ub])
            # Clip grads below. LB should ++. Thus grad for LB should < 0 (LB' = LB - lr * grad). Similar for UB.
            grads[0].clamp_(max=0.)  # LB
            grads[1].clamp_(min=0.)  # UB
            new_grad = sum([g.abs() for g in grads])  # get one value for each abstraction
        return new_grad, new_safe_dist

    def _grad_dists_of(self, new_lb: Tensor, new_ub: Tensor, new_extra: Optional[Tensor], forward_fn: nn.Module,
                       batch_size: int) -> Tuple[Tensor, Tensor]:
        """ Dispatch the computation to be batch-by-batch.
        :param batch_size: compute the gradients batch-by-batch, so as to avoid huge memory consumption at once.
        """
        absset = AbsData(new_lb, new_ub, new_extra)
        abs_loader = DataLoader(absset, batch_size=batch_size, shuffle=False)

        split_grads, split_safe_dists = [], []
        for batch in abs_loader:
            if new_extra is None:
                batch_lb, batch_ub = batch
                batch_extra = None
            else:
                batch_lb, batch_ub, batch_extra = batch
            new_grad, new_safe_dist = self._grad_dists_of_batch(batch_lb, batch_ub, batch_extra, forward_fn)
            split_grads.append(new_grad)
            split_safe_dists.append(new_safe_dist)

        split_grads = torch.cat(split_grads, dim=0)
        split_safe_dists = torch.cat(split_safe_dists, dim=0)
        return split_grads, split_safe_dists

    @staticmethod
    def _transfer_safe(new_lb: Tensor, new_ub: Tensor, new_extra: Optional[Tensor], new_safe_dist: Tensor,
                       new_grad: Tensor) -> Tuple[Tuple[Tensor, Tensor, Optional[Tensor]],
                                                  Tuple[Tensor, Tensor, Optional[Tensor], Tensor, Tensor]]:
        safe_bits = new_safe_dist <= 0.
        rem_bits = ~ safe_bits

        new_safe_lb, rem_lb = new_lb[safe_bits], new_lb[rem_bits]
        new_safe_ub, rem_ub = new_ub[safe_bits], new_ub[rem_bits]
        new_safe_extra = None if new_extra is None else new_extra[safe_bits]
        rem_extra = None if new_extra is None else new_extra[rem_bits]
        rem_safe_dist, rem_grad = new_safe_dist[rem_bits], new_grad[rem_bits]

        return (new_safe_lb, new_safe_ub, new_safe_extra),\
               (rem_lb, rem_ub, rem_extra, rem_safe_dist, rem_grad)

    @staticmethod
    def _pick_top(top_k: int, wl_lb: Tensor, wl_ub: Tensor, wl_extra: Optional[Tensor], wl_safe_dist: Tensor,
                  wl_grad: Tensor, largest: bool) -> Tuple[Tensor, Tensor, Optional[Tensor], Tensor,
                                                           Tensor, Tensor, Optional[Tensor], Tensor, Tensor]:
        """ Use safety loss to pick the abstractions among current work list for bisection.
        :param largest: either pick the largest safety loss or smallest safety loss
        :return: the separated parts of LB, UB, extra (if any), grad, safe_dist
        """
        top_k = min(top_k, len(wl_lb))

        ''' If pytorch has mergesort(), do that, right now it's just simpler (and faster) to use topk().
            I also tried the heapq module, adding each unbatched tensor to heap. However, That is much slower and
            heappop() raises some weird RuntimeError: Boolean value of Tensor with more than one value is ambiguous..
        '''
        _, topk_idxs = wl_safe_dist.topk(top_k, largest=largest, sorted=False)  # topk_idxs: size <K>
        # scatter, topk_idxs are 0, others are 1
        other_idxs = torch.ones(len(wl_safe_dist), device=wl_safe_dist.device).byte().scatter_(-1, topk_idxs, 0)
        other_idxs = other_idxs.nonzero(as_tuple=True)  # <Batch-K>

        batch_lb, rem_lb = wl_lb[topk_idxs], wl_lb[other_idxs]
        batch_ub, rem_ub = wl_ub[topk_idxs], wl_ub[other_idxs]
        batch_extra = None if wl_extra is None else wl_extra[topk_idxs]
        rem_extra = None if wl_extra is None else wl_extra[other_idxs]
        rem_safe_dist = wl_safe_dist[other_idxs]  # batch_safe_dist is no longer needed, refinement only needs the grad
        batch_grad, rem_grad = wl_grad[topk_idxs], wl_grad[other_idxs]

        return batch_lb, batch_ub, batch_extra, batch_grad,\
               rem_lb, rem_ub, rem_extra, rem_grad, rem_safe_dist

    def _sample_check(self, lb: Tensor, ub: Tensor, extra: Optional[Tensor], forward_fn: nn.Module) -> Optional[Tensor]:
        """ Sample points from given input abstractions and check their safety. If found cex, return it/them. """
        # just sample 1 point per abstraction for now, can have K>1 if extra is also duplicated
        pts = sample_points(lb, ub, K=1)
        outs = forward_fn(pts)
        viol_dist = self.prop.viol_dist_conc(outs, extra)
        viol_bits = viol_dist <= 0.
        if viol_bits.any():
            cex = pts[viol_bits]
            return cex
        return None

    def verify(self, lb: Tensor, ub: Tensor, extra: Optional[Tensor], forward_fn: nn.Module,
               batch_size: int = 4096) -> Optional[Tensor]:
        """ Verify the safety property or return some found counterexamples.

            The major difference with split() is that verify() does depth-first-search, checking smaller loss
            abstractions first. Otherwise, the memory consumption of BFS style refinement will explode.

            Also, tiny_width is not considered in verify(), it aims to enumerate however small areas, anyway.

        :param lb: Batch x ...
        :param ub: Batch x ...
        :param extra: could contain extra info such as the bit vectors for each LB/UB cube showing which safety property
                      it should satisfy in AndProp; or just None
        :param forward_fn: differentiable forward propagation, not passing in net and call net(input) because different
                           applications may have different net(input, **kwargs)
        :param batch_size: how many to bisect once at one time
        :return: (batched) counterexample tensors, if not None
        """
        assert valid_lb_ub(lb, ub)
        assert batch_size > 0

        # track how much have been certified
        tot_area = total_area(lb, ub)
        assert tot_area > 0
        safes_area = 0.
        t0 = timer()

        def empty() -> Tensor:
            return empty_like(lb)

        # no need to save safe_lb/safe_ub
        wl_lb, wl_ub = empty(), empty()
        wl_extra = None if extra is None else empty().byte()
        wl_safe_dist, wl_grad = empty(), empty()

        new_lb, new_ub, new_extra = lb, ub, extra
        iter = 0
        while True:
            iter += 1

            if len(new_lb) > 0:
                ''' It's important to have no_grad() here, otherwise the GPU memory will keep growing. With no_grad(),
                    the GPU memory usage is stable. enable_grad() is called inside for grad computation.

                    viol_dist is now removed, because if viol_dist can certify violation, sampling can absolutely do the
                    same, vice NOT versa. So there is no need to compute viol_dist anymore. It also shows that using
                    'safe' is slightly better than 'viol' as source based on first two hard instances in acas-hard.

                    I also tried using a 'factor' tensor before, with LB = LB * factor and UB = UB * factor, to compute
                    gradient w.r.t. 'factor'. However, that is much worse than the grad w.r.t. LB and UB directly. One
                    possible reason is that 'factor' can only shrink the space in one direction towards its mid point.
                    This has little to do with actual bisection later on. Grads w.r.t. LB/UB is more directly related.
                '''
                with torch.no_grad():
                    new_grad, new_safe_dist = self._grad_dists_of(new_lb, new_ub, new_extra, forward_fn, batch_size)

                logging.debug(f'At iter {iter}, another {len(new_lb)} boxes are processed.')

                # process safe abstractions here rather than later
                (new_safe_lb, new_safe_ub, _), (rem_lb, rem_ub, rem_extra, rem_safe_dist, rem_grad) =\
                    self._transfer_safe(new_lb, new_ub, new_extra, new_safe_dist, new_grad)
                logging.debug(f'In which {len(new_safe_lb)} confirmed safe.')

                new_safes_area = total_area(new_safe_lb, new_safe_ub)
                safes_area += new_safes_area

                if len(rem_lb) > 0:
                    # sample check the rest and add to worklist
                    cex = self._sample_check(rem_lb, rem_ub, rem_extra, forward_fn)
                    if cex is not None:
                        # found cex!
                        logging.debug(f'CEX found by sampling: {cex}')
                        return cex

                wl_lb = cat0(wl_lb, rem_lb)
                wl_ub = cat0(wl_ub, rem_ub)
                wl_extra = cat0(wl_extra, rem_extra)
                wl_safe_dist = cat0(wl_safe_dist, rem_safe_dist)
                wl_grad = cat0(wl_grad, rem_grad)

            safe_area_percent = safes_area / tot_area * 100
            wl_area_percent = 100. - safe_area_percent
            logging.debug(f'After iter {iter}, {pp_time(timer() - t0)}, total ({safe_area_percent:.2f}%) safe, ' +
                          f'total #{len(wl_lb)} ({wl_area_percent:.2f}%) in worklist.')
            # logging.debug(pp_cuda_mem())

            if len(wl_lb) == 0:
                # nothing to bisect anymore
                break

            logging.debug(f'In worklist, safe dist min: {wl_safe_dist.min()}, max: {wl_safe_dist.max()}.')

            ''' Pick small loss boxes to bisect first for verification, otherwise BFS style consumes huge memory.
                There is no need to check if entire wl is selected, topk() should do that automatically (I suppose).
            '''
            tmp = self._pick_top(batch_size, wl_lb, wl_ub, wl_extra, wl_safe_dist, wl_grad, largest=False)
            batch_lb, batch_ub, batch_extra, batch_grad = tmp[:4]
            wl_lb, wl_ub, wl_extra, wl_grad, wl_safe_dist = tmp[4:]

            new_lb, new_ub, new_extra = by_smear(batch_lb, batch_ub, batch_extra, batch_grad)
        return None

    @staticmethod
    def _transfer_tiny(new_lb: Tensor, new_ub: Tensor, new_extra: Optional[Tensor],
                       new_safe_dist: Tensor, new_grad: Tensor,
                       tiny_width: float) -> Tuple[Tuple[Tensor, Tensor, Optional[Tensor]],
                                                   Tuple[Tensor, Tensor, Optional[Tensor], Tensor, Tensor]]:
        width = new_ub - new_lb
        tiny_bits = (width <= tiny_width).all(dim=1)  # pick those whose dimensions are all tiny
        rem_bits = ~ tiny_bits

        new_tiny_lb, rem_lb = new_lb[tiny_bits], new_lb[rem_bits]
        new_tiny_ub, rem_ub = new_ub[tiny_bits], new_ub[rem_bits]
        new_tiny_extra = None if new_extra is None else new_extra[tiny_bits]
        rem_extra = None if new_extra is None else new_extra[rem_bits]
        rem_safe_dist, rem_grad = new_safe_dist[rem_bits], new_grad[rem_bits]

        return (new_tiny_lb, new_tiny_ub, new_tiny_extra),\
               (rem_lb, rem_ub, rem_extra, rem_safe_dist, rem_grad)

    def split(self, lb: Tensor, ub: Tensor, extra: Optional[Tensor], forward_fn: nn.Module, batch_size: int,
              stop_on_k_all: int = None,
              stop_on_k_new: int = None,
              stop_on_k_ops: int = None,
              tiny_width: float = None,
              collapse_res: bool = True) -> Union[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor, Tensor]]:
        """ Different from verify(), split() does breadth-first traversal. Its objective is to have roughly even
            abstractions with small safety losses for the optimization later.

        :param lb: could be accessed from props.lbub(), but may need additional normalization
        :param ub: same as @param lb
        :param extra: could contain extra info such as the bit vectors for each LB/UB cube showing which safety property
                      it should satisfy in AndProp; or just None
        :param forward_fn: differentiable forward propagation, not passing in net and call net(input) because different
                           applications may have different net(input, **kwargs)
        :param batch_size: How many to bisect once at most, must provide this granularity.
                           Larger batch_size => faster to compute but less precise / averaged (due to more rushing).
        :param stop_on_k_all: if not None, split() stops after total amount of abstractions exceeds this bar.
        :param stop_on_k_new: if not None, split() stops after the amount of abstractions introduced by this split()
                              call exceeds this bar.
        :param stop_on_k_ops: if not None, split() stops after this many refinement steps have been applied.
        :param tiny_width: if not None, stop refining one dimension if its width is already <= this bar,
                           e.g., setting tiny_width=1e-3 would ensure all refined abstraction dimension width > 5e-4.
        :return: <LB, UB> when extra is None, otherwise <LB, UB, extra>
        """
        assert valid_lb_ub(lb, ub)
        assert batch_size > 0

        def _validate_stop_criterion(v, pivot: int):
            assert v is None or (isinstance(v, int) and v > pivot)
            return
        _validate_stop_criterion(stop_on_k_all, 0)
        _validate_stop_criterion(stop_on_k_new, 0)
        _validate_stop_criterion(stop_on_k_ops, -1)  # allow 0 refinement steps, i.e., just evaluate, no refine

        def empty() -> Tensor:
            return empty_like(lb)

        n_orig_abs = len(lb)

        # Not storing viol_lb anymore, as viol_dist is no longer computed. Those violated regions are still refined.
        wl_lb, wl_ub = empty(), empty()
        safe_lb, safe_ub = empty(), empty()
        tiny_lb, tiny_ub = empty(), empty()
        wl_extra = None if extra is None else empty().byte()
        safe_extra = None if extra is None else empty().byte()
        tiny_extra = None if extra is None else empty().byte()
        wl_safe_dist, wl_grad = empty(), empty()

        new_lb, new_ub, new_extra = lb, ub, extra
        iter = 0
        while True:
            iter += 1

            if len(new_lb) > 0:
                with torch.no_grad():
                    ''' It's important to have no_grad() here, otherwise the GPU memory will keep growing. With
                        no_grad(), the GPU memory usage is stable. enable_grad() is called inside for grad computation.
                    '''
                    new_grad, new_safe_dist = self._grad_dists_of(new_lb, new_ub, new_extra, forward_fn, batch_size)

                logging.debug(f'At iter {iter}, another {len(new_lb)} boxes are processed.')

                # process safe abstractions here rather than later
                (new_safe_lb, new_safe_ub, new_safe_extra), (rem_lb, rem_ub, rem_extra, rem_safe_dist, rem_grad) =\
                    self._transfer_safe(new_lb, new_ub, new_extra, new_safe_dist, new_grad)
                logging.debug(f'In which {len(new_safe_lb)} confirmed safe.')

                safe_lb = cat0(safe_lb, new_safe_lb)
                safe_ub = cat0(safe_ub, new_safe_ub)
                safe_extra = cat0(safe_extra, new_safe_extra)

                if tiny_width is not None:
                    (new_tiny_lb, new_tiny_ub, new_tiny_extra), (rem_lb, rem_ub, rem_extra, rem_safe_dist, rem_grad) =\
                        self._transfer_tiny(rem_lb, rem_ub, rem_extra, rem_safe_dist, rem_grad, tiny_width)
                    tiny_lb = cat0(tiny_lb, new_tiny_lb)
                    tiny_ub = cat0(tiny_ub, new_tiny_ub)
                    tiny_extra = cat0(tiny_extra, new_tiny_extra)
                    logging.debug(f'In which {len(new_tiny_lb)} confirmed tiny.')

                wl_lb = cat0(wl_lb, rem_lb)
                wl_ub = cat0(wl_ub, rem_ub)
                wl_extra = cat0(wl_extra, rem_extra)
                wl_safe_dist = cat0(wl_safe_dist, rem_safe_dist)
                wl_grad = cat0(wl_grad, rem_grad)

            logging.debug(f'After iter {iter}, total #{len(safe_lb)} safe, #{len(wl_lb)} in worklist, ' +
                          f'total #{len(tiny_lb)} too small and ignored.')

            if len(wl_lb) == 0:
                # nothing to bisect anymore
                break

            logging.debug(f'At iter {iter}, worklist safe dist min: {wl_safe_dist.min()}, max: {wl_safe_dist.max()}.')

            n_curr_abs = len(safe_lb) + len(tiny_lb) + len(wl_lb)
            if stop_on_k_all is not None and n_curr_abs >= stop_on_k_all:
                # has collected enough abstractions
                break
            if stop_on_k_new is not None and n_curr_abs - n_orig_abs >= stop_on_k_new:
                # has collected enough new abstractions
                break
            if stop_on_k_ops is not None and iter > stop_on_k_ops:
                # has run enough refinement iterations
                break

            ''' Pick large loss boxes to bisect first for splitting, so as to generate evenly distributed areas.
                There is no need to check if entire wl is selected, topk() should do that automatically (I suppose).
            '''
            tmp = self._pick_top(batch_size, wl_lb, wl_ub, wl_extra, wl_safe_dist, wl_grad, largest=True)
            batch_lb, batch_ub, batch_extra, batch_grad = tmp[:4]
            wl_lb, wl_ub, wl_extra, wl_grad, wl_safe_dist = tmp[4:]

            new_lb, new_ub, new_extra = by_smear(batch_lb, batch_ub, batch_extra, batch_grad)
            pass  # end of worklist while

        logging.debug(f'\nAt the end, split {len(wl_lb)} uncertain (non-zero loss) boxes, ' +
                      f'{len(safe_lb)} safe boxes and {len(tiny_lb)} tiny boxes.')
        if len(wl_lb) > 0:
            logging.debug(f'Non zero loss boxes have safe loss min {wl_safe_dist.min()} ~ max {wl_safe_dist.max()}.')

        if collapse_res:
            with torch.no_grad():
                all_lb = cat0(wl_lb, safe_lb, tiny_lb)
                all_ub = cat0(wl_ub, safe_ub, tiny_ub)
                all_extra = cat0(wl_extra, safe_extra, tiny_extra)

            if all_extra is None:
                return all_lb, all_ub
            else:
                return all_lb, all_ub, all_extra
        else:
            with torch.no_grad():
                wl_lb = cat0(wl_lb, tiny_lb)
                wl_ub = cat0(wl_ub, tiny_ub)
                wl_extra = cat0(wl_extra, tiny_extra)
            if wl_extra is None:
                return wl_lb, wl_ub
            else:
                return wl_lb, wl_ub, wl_extra

    def try_certify(self, lb: Tensor, ub: Tensor, extra: Optional[Tensor], forward_fn: nn.Module, batch_size: int,
              timeout_sec: int) -> bool:
        """
        :return: True if it can successfully certify the property on lb/ub within certain limits
        """
        try:
            with timeout(sec=timeout_sec):
                cex = self.verify(lb, ub, extra, forward_fn, batch_size)
            return cex is None
        except TimeoutError:
            logging.info(f'try_certify() time out before certified or falsified after {timeout_sec} seconds.')
            return False
    pass


def bisect_by(lb: Tensor, ub: Tensor, idxs: Tensor,
              extra: Tensor = None) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
    """ Bisect specific columns.
    :param idxs: <Batch>, as the indices from torch.max()
    :param extra: if not None, it contains the bit vector for each LB/UB piece showing which prop they should obey
    """
    # scatter_() to convert indices into one-hot encoding
    split_idxs = idxs.unsqueeze(dim=-1)  # Batch x 1
    onehot_idxs = torch.zeros_like(lb).byte().scatter_(-1, split_idxs, 1)

    # then bisect the specified cols only
    mid = (lb + ub) / 2.0
    lefts_lb = lb
    lefts_ub = torch.where(onehot_idxs, mid, ub)
    rights_lb = torch.where(onehot_idxs, mid, lb)
    rights_ub = ub

    newlb = cat0(lefts_lb, rights_lb)
    newub = cat0(lefts_ub, rights_ub)
    newextra = cat0(extra, extra)
    return newlb, newub, newextra


def by_smear(new_rem_lb: Tensor, new_rem_ub: Tensor, new_rem_extra: Optional[Tensor], new_rem_grad: Tensor,
             tiny_width: float = None) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
    """ Experiment shows that smear = grad * dim_width as in ReluVal is the best heuristic tried so far. It's better
        than either one alone, and better than other indirect loss e.g., introduced over-approximated area. Also tried
        K-smear-followed-by-1-width, or grad-then-width, both become worse. Moreover, normalization on grad/width also
        makes it worse.
    """
    with torch.no_grad():
        width = new_rem_ub - new_rem_lb
        assert new_rem_lb.dim() == 2, 'Otherwise, I need to reduce the >2 dims to compute dim width?'
        smears = new_rem_grad * width  # tried normalization, didn't do any better..

        if tiny_width is not None:
            # consider only those dimensions that are not tiny
            not_tiny_bits = width > tiny_width
            smears = smears * not_tiny_bits.float()

        _, split_idxs = smears.max(dim=-1)
        return bisect_by(new_rem_lb, new_rem_ub, split_idxs, new_rem_extra)


def by_width(new_rem_lb: Tensor, new_rem_ub: Tensor, new_rem_extra: Optional[Tensor], new_rem_grad: Tensor,
             tiny_width: float = None) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
    width = new_rem_ub - new_rem_lb
    assert new_rem_lb.dim() == 2, 'Otherwise, I need to reduce the >2 dims to compute dim width?'
    split_idxs = width.argmax(dim=-1)
    return bisect_by(new_rem_lb, new_rem_ub, split_idxs, new_rem_extra)
