""" Refinement guided by concrete samples based clustering. """

import logging
import sys
from pathlib import Path
from typing import Tuple, Optional, Union, List
from timeit import default_timer as timer

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from sklearn.cluster import AgglomerativeClustering
from sklearn import tree

from diffabs import AbsDom, AbsData
from diffabs.utils import valid_lb_ub

sys.path.append(str(Path(__file__).resolve().parent.parent))

from art.prop import AbsProp
from art.utils import total_area, pp_time, gen_vtx_points


def empty_like(t: Tensor) -> Tensor:
    """ Empty tensor (different from torch.empty() for concatenation. """
    return torch.tensor([], device=t.device)


def cat0(*ts: Tensor) -> Optional[Tensor]:
    """ Usage: simplify `torch.cat((ts1, ts2), dim=0)` to `cat0(ts1, ts2)`. """
    if ts[0] is None:
        return None
    return torch.cat(ts, dim=0)


class Cluster(object):
    """ Instead of bisection, we apply clustering-based K-splitting at once.

        In previous approach (bisecter.py), we refine input abstractions by bisecting one into two using gradient-based
        heuristic. Such method has several disadvantages:
        (1) Bisecting generates only one more abstraction along one dimension at a time. To generate more abstractions,
            it needs to pass the input abstraction through abstract domain again, which is slow. This is also getting
            harder in high dimensional problems, such as images.
        (2) Information revealed by gradients may not help much, especially when (a) the widths are already small, as
            shown in `tests/inspect_refinement_choices.py`; or (b) they are accumulated throughout multiple steps in
            CPS. It suffers from this issue more than normal RL because running on abstract domains generates less
            precise gradients.
        (3) When the certification fails, there are few feedback on whether there exists a counterexample or just due to
            over-approximation. The newer version of Bisecter samples 1 point to check violation. It can be extended to
            sample K points per abstraction. But that is not a principled way, just random sampling.

        To address these issues, we propose the K-Clustering based refinement technique, with the key intuition that
        those input points "geometrically close" and "property equivalent" in the output space should be categorized in
        the same input abstraction. After all, the abstract domains we use are geometric. Therefore, our basic algorithm
        is shown below. I tried to inherit from some common base class of Bisecter, but that is not necessary.

        (1) Given input abstractions, filter out those certified ones;
        (2) For each uncertified input abstraction, generate grid points and compute their corresponding outputs;
        (3) If any safety violation is observed, report it and terminate;
        (4) Apply clustering algorithm on the output points;
        (5) Refine input abstractions according to the output clusters, using e.g., Decision Tree constructed rules.
            Safety violations are in different clusters.
        Or (4-5) cluster input abstractions directly based on output states;
        (6) Redo steps (1)-(5) until a) all are certified and/or b) all abstractions are small enough and/or
            c) maximum amount exceeded.

        TODO Future Work:
        *   Replace interval/box shaped DT construction with some linear regression based separator. So as to introduce
            new base variable based abstraction. This improves DP domain.
    """

    def __init__(self, dom: AbsDom, prop: AbsProp):
        """
        :param dom: the abstract domain module to use
        """
        self.d = dom
        self.prop = prop
        return

    def _dists_of(self, new_lb: Tensor, new_ub: Tensor, new_extra: Optional[Tensor], forward_fn: nn.Module,
                  batch_size: int) -> Tensor:
        """ Dispatch the computation to be batch-by-batch.
        :param batch_size: compute the gradients batch-by-batch, so as to avoid huge memory consumption at once.
        """
        absset = AbsData(new_lb, new_ub, new_extra)
        abs_loader = DataLoader(absset, batch_size=batch_size, shuffle=False)

        split_safe_dists = []
        for batch in abs_loader:
            if new_extra is None:
                batch_lb, batch_ub = batch
                batch_extra = None
            else:
                batch_lb, batch_ub, batch_extra = batch

            batch_ins = self.d.Ele.by_intvl(batch_lb, batch_ub)
            batch_outs = forward_fn(batch_ins)
            new_safe_dist = self.prop.safe_dist(batch_outs) if batch_extra is None\
                else self.prop.safe_dist(batch_outs, batch_extra)
            split_safe_dists.append(new_safe_dist)

        split_safe_dists = torch.cat(split_safe_dists, dim=0)
        return split_safe_dists

    @staticmethod
    def _transfer_safe(new_lb: Tensor, new_ub: Tensor, new_extra: Optional[Tensor], new_safe_dist: Tensor) ->\
            Tuple[Tuple[Tensor, Tensor, Optional[Tensor]],
                  Tuple[Tensor, Tensor, Optional[Tensor], Tensor]]:
        safe_bits = new_safe_dist <= 0.
        rem_bits = ~ safe_bits

        new_safe_lb, rem_lb = new_lb[safe_bits], new_lb[rem_bits]
        new_safe_ub, rem_ub = new_ub[safe_bits], new_ub[rem_bits]
        new_safe_extra = None if new_extra is None else new_extra[safe_bits]
        rem_extra = None if new_extra is None else new_extra[rem_bits]
        rem_safe_dist = new_safe_dist[rem_bits]

        return (new_safe_lb, new_safe_ub, new_safe_extra),\
               (rem_lb, rem_ub, rem_extra, rem_safe_dist)

    @staticmethod
    def _pick_top(top_k: int, wl_lb: Tensor, wl_ub: Tensor, wl_extra: Optional[Tensor], wl_safe_dist: Tensor,
                  largest: bool) -> Tuple[Tensor, Tensor, Optional[Tensor],
                                          Tensor, Tensor, Optional[Tensor], Tensor]:
        """ Use safety loss to pick the abstractions among current work list for bisection.
        :param largest: either pick the largest safety loss or smallest safety loss
        :return: the separated parts of LB, UB, extra (if any), safe_dist
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

        return batch_lb, batch_ub, batch_extra,\
               rem_lb, rem_ub, rem_extra, rem_safe_dist

    def verify(self, lb: Tensor, ub: Tensor, extra: Optional[Tensor], forward_fn: nn.Module,
               batch_size: int = 200) -> Optional[Tensor]:
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
        :param batch_size: how many abstractions are checked safe at a time
        :param sample_size: how many points are sampled per abstraction for refinement
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
        wl_safe_dist = empty()

        new_lb, new_ub, new_extra = lb, ub, extra
        iter = 0
        while True:
            iter += 1

            if len(new_lb) > 0:
                ''' It's important to have no_grad() here, otherwise the GPU memory will keep growing. With no_grad(),
                    the GPU memory usage is stable. enable_grad() is called inside for grad computation.
                '''
                with torch.no_grad():
                    new_safe_dist = self._dists_of(new_lb, new_ub, new_extra, forward_fn, batch_size)

                logging.debug(f'At iter {iter}, another {len(new_lb)} boxes are processed.')

                # process safe abstractions here rather than later
                (new_safe_lb, new_safe_ub, _), (rem_lb, rem_ub, rem_extra, rem_safe_dist) =\
                    self._transfer_safe(new_lb, new_ub, new_extra, new_safe_dist)
                logging.debug(f'In which {len(new_safe_lb)} confirmed safe.')

                new_safes_area = total_area(new_safe_lb, new_safe_ub)
                safes_area += new_safes_area

                ''' It was sampling to check cex here, right after processing new abstractions in bisecter.py,
                    here the sampling can be left later until the sampling for clustering.
                '''

                wl_lb = cat0(wl_lb, rem_lb)
                wl_ub = cat0(wl_ub, rem_ub)
                wl_extra = cat0(wl_extra, rem_extra)
                wl_safe_dist = cat0(wl_safe_dist, rem_safe_dist)

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
            tmp = self._pick_top(batch_size, wl_lb, wl_ub, wl_extra, wl_safe_dist, largest=False)
            batch_lb, batch_ub, batch_extra = tmp[:3]
            wl_lb, wl_ub, wl_extra, wl_safe_dist = tmp[3:]

            # refine these batch_lb/ubs
            ''' One alternative is to generate grid points using torch.linspace() and/or torch.meshgrid(). But that can
                only generate one meshgrid for one abstraction at a time, which should be slower than the batched
                version of generating random points for all given abstractions at once.

                Moreover, the random points way also allows directly control on how many points per abstraction are
                sampled. If using meshgrid, the exact number of sampled points are growing exponentially as the
                dimension increases.
            '''
            # sampled_pts, sampled_extra = gen_rnd_points(batch_lb, batch_ub, batch_extra, K=sample_size)
            sampled_pts, sampled_extra = gen_vtx_points(batch_lb, batch_ub, batch_extra)  # faster using vertices

            logging.debug(f'From {len(batch_lb)} abstractions, sampled points shape {sampled_pts.shape}.')
            with torch.no_grad():
                sampled_outs = forward_fn(sampled_pts)

            # check cex from sampled points
            old_shape = list(sampled_outs.shape)
            viol_dist = self.prop.viol_dist_conc(sampled_outs.flatten(0, 1), sampled_extra)

            viol_bits = viol_dist <= 0.
            if viol_bits.any():
                cex = sampled_pts[viol_bits]
                logging.debug(f'CEX found by sampling: {cex}')
                cex = cex.flatten(0, 1)  # Batch x K x States => (Batch * K) x States
                return cex

            sampled_outs = sampled_outs.view(*old_shape)
            tmp_t0 = timer()
            refined_outs = self.by_clustering(batch_lb, batch_ub, batch_extra, sampled_pts, sampled_outs)
            logging.debug(f'Refinement in total takes {pp_time(timer() - tmp_t0)}')
            new_lb, new_ub = refined_outs[:2]
            new_extra = None if batch_extra is None else refined_outs[2]
        return None

    def by_clustering(self, batch_lb: Tensor, batch_ub: Tensor, batch_extra: Optional[Tensor], sampled_pts: Tensor,
                      sampled_outs: Tensor) -> Union[Tuple[Tensor, Tensor],
                                                     Tuple[Tensor, Tensor, Tensor]]:
        """ Refine by clustering. The corresponding points in the output space have been computed. Now apply clustering
            to all of them. Algorithm:
            (1) Cluster the sampled outputs and collect their labels;
            (2) Construct a decision tree to divide the sampled inputs according to the clustered labels;
            (3) Refine input abstractions according to the rules of the new decision tree.

        :param batch_lb: batched LBs
        :param batch_ub: batched UBs
        :param batch_extra: batched Extras, if not None
        :param sampled_pts: Batch x K x State
        :param sampled_outs: Batch x K x State
        :return: if extra is None, return <refined LB, UB> without extra, otherwise return with extra
        """
        sampled_pts = sampled_pts.cpu().numpy()

        ''' Results show that normalization does make some difference, improves a bit, even if clustering should not
            care too much about absolute value change.. Normalization overall rather than per-abstraction is not worse
            and faster.
        '''
        std, mean = torch.std_mean(sampled_outs)
        sampled_outs = (sampled_outs - mean) / std
        sampled_outs = sampled_outs.cpu().numpy()

        all_lbs, all_ubs = [], []
        all_extras = None if batch_extra is None else []

        cluster_secs, dt_secs, split_secs = 0., 0., 0.
        for i in range(len(batch_lb)):
            lb, ub, pts, outs = batch_lb[i], batch_ub[i], sampled_pts[i], sampled_outs[i]
            extra = None if batch_extra is None else batch_extra[i]

            ''' (1) Cluster the sampled outputs and collect their labels, per abstraction. I tried to cluster all points
                altogether and then construct a decision tree per abstraction regarding only in-abstraction labels. But
                that turns out too slow, due to the large amount of points being considered. That makes the overall
                algorithm absolutely impractical.

                Ward / AgglomerativeClustering is chosen after some runs, because it shows fastest among all provided
                methods from https://scikit-learn.org/stable/modules/clustering.html. Other methods are either too slow
                (specifically, KMeans, due to many init runs) or do not refine at all.
            '''
            c = AgglomerativeClustering(n_clusters=2)  # cluster into 2 parts, decision tree will extract >2 rules later
            t0 = timer()
            labels = c.fit_predict(outs)
            cluster_secs += (timer() - t0)

            ''' (2) Construct a decision tree to divide the sampled inputs according to the clustered labels. '''
            dt = tree.DecisionTreeClassifier(max_leaf_nodes=2)  # forcing to bisect actually improves the result
            t0 = timer()
            dt.fit(pts, labels)
            dt_secs += (timer() - t0)

            # features = [f'{i}-{v.name}' for i, v in enumerate(acas.AcasIn)]
            # assert len(outs[0]) == len(features)
            # tree.export_graphviz(dt, out_file='dt-output.gv', feature_names=features, node_ids=True)
            # sp.check_call(['dot', '-T', 'png', 'dt-output.gv', '-o', 'dt-output.png'])
            # exit(0)

            ''' (3) Refine input abstractions according to the rules of the new decision tree. According to
                https://scikit-learn.org/stable/modules/tree.html#tree-algorithms-id3-c4-5-c5-0-and-cart, CART is
                implemented in Scikit-Learn, which does not compute rule sets. So we have to reconstruct the splitting
                rules by hand.
            '''
            # One potential improvement is to collect the DT rules (when max_leaf_nodes=2) for all, then bisect by batch
            t0 = timer()
            split_lbs, split_ubs, split_extras = self._split_by_dt_rule(lb, ub, extra, dt)
            split_secs += (timer() - t0)
            all_lbs.extend(split_lbs)
            all_ubs.extend(split_ubs)
            if all_extras is not None:
                all_extras.append(split_extras)  # already batched
            pass  # end of refining each input abstraction

        logging.debug(f'In total, clustering takes {pp_time(cluster_secs)}, ' +
                      f'decision tree takes {pp_time(dt_secs)}, ' +
                      f'splitting takes {pp_time(split_secs)}.')

        all_lbs = torch.stack(all_lbs, dim=0)
        all_ubs = torch.stack(all_ubs, dim=0)
        if all_extras is not None:
            all_extras = torch.cat(all_extras, dim=0)

        if all_extras is None:
            return all_lbs, all_ubs
        else:
            return all_lbs, all_ubs, all_extras

    @staticmethod
    def _split_by_dt_rule(lb: Tensor, ub: Tensor, extra: Optional[Tensor],
                          dt: tree.DecisionTreeClassifier) -> Tuple[List[Tensor], List[Tensor], Optional[Tensor]]:
        """ Recursively iterate the constructed decision tree and generate refined input abstractions accordingly.
        :param lb/ub/extra: unbatched
        :return: Lists, can merge into batched Tensor later on. Note that extra is not a list, because it they are all
                 the same per input abstraction, so just expand().
        """
        split_lbs, split_ubs = [], []
        tr = dt.tree_

        def _visit(node, curr_lb: Tensor, curr_ub: Tensor):
            assert node != tree._tree.TREE_LEAF
            if tr.feature[node] == tree._tree.TREE_UNDEFINED:
                # leaf
                split_lbs.append(curr_lb)
                split_ubs.append(curr_ub)
                return

            feature = tr.feature[node]
            threshold = tr.threshold[node]

            assert curr_lb[feature] <= threshold <= curr_ub[feature]
            left_lb, right_lb = curr_lb.clone(), curr_lb.clone()
            left_ub, right_ub = curr_ub.clone(), curr_ub.clone()
            left_ub[feature] = threshold
            right_lb[feature] = threshold

            _visit(tr.children_left[node], left_lb, left_ub)
            _visit(tr.children_right[node], right_lb, right_ub)
            return

        _visit(0, lb, ub)  # 0 is the root node

        if extra is None:
            split_extra = None
        else:
            shape = list(extra.shape)
            shape.insert(0, len(split_lbs))
            split_extra = extra.expand(*shape)
        return split_lbs, split_ubs, split_extra
    pass
