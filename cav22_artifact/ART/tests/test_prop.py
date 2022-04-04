""" Validating the algorithms of lbub exclusion and AndProp join. """

import sys
from pathlib import Path

import torch

from diffabs import DeeppolyDom

sys.path.append(str(Path(__file__).resolve().parent.parent))

from art.prop import lbub_exclude, AndProp
from art.acas import AcasNetID
from art.utils import total_area


def test_lbub_exclusion_1():
    """ Validate that LB/UB intersection/exclusion ops are correct. """
    lb1, ub1 = torch.Tensor([1, 1]), torch.Tensor([4, 4])
    lb2, ub2 = torch.Tensor([2, 2]), torch.Tensor([3, 3])
    res_lb, res_ub = lbub_exclude(lb1, ub1, lb2, ub2)

    # each dimension adds 2 pieces (left & right), no overlapping
    assert len(res_lb) == len(res_ub)
    assert len(res_lb) == 2 * len(lb1)
    return


def test_lbub_exclusion_2():
    lb1, ub1 = torch.Tensor([1, 1]), torch.Tensor([4, 4])
    lb2, ub2 = torch.Tensor([2, 2]), torch.Tensor([3, 4])
    res_lb, res_ub = lbub_exclude(lb1, ub1, lb2, ub2)

    # overlapped on one dimension
    assert len(res_lb) == len(res_ub)
    assert len(res_lb) == 2 * (len(lb1) - 1) + 1
    return


def test_lbub_exclusion_3():
    lb1, ub1 = torch.Tensor([1, 1, 1]), torch.Tensor([4, 4, 4])
    lb2, ub2 = torch.Tensor([2, 2, 2]), torch.Tensor([3, 3, 3])
    res_lb, res_ub = lbub_exclude(lb1, ub1, lb2, ub2)
    # each dimension adds 2 pieces (left & right), no overlapping
    assert len(res_lb) == len(res_ub)
    assert len(res_lb) == 2 * len(lb1)
    return


def test_andprop_conjoin():
    """ Validate (manually..) that the AndProp is correct. """
    dom = DeeppolyDom()

    def _go(id):
        props = id.applicable_props(dom)
        ap = AndProp(props)

        print('-- For network', id)
        for p in props:
            print('-- Has', p.name)
            lb, ub = p.lbub()
            print('   LB:', lb)
            print('   UB:', ub)

        lb, ub = ap.lbub()
        print('-- All conjoined,', ap.name)
        print('   LB:', lb)
        print('   UB:', ub)
        print('   Labels:', ap.labels)
        print('Cnt:', len(lb))
        for i in range(len(lb)):
            print('  ', i, 'th piece, width:', ub[i] - lb[i], f'area: {total_area(lb[[i]], ub[[i]]) :E}')
        print()
        return

    ''' <1, 1> is tricky, as it has many props;
        <1, 9> is special, as it is different from many others;
        Many others have prop1, prop2, prop3, prop4 would generate 3 pieces, in which prop1 and prop2 merged.
    '''
    # _go(AcasNetID(1, 1))
    # _go(AcasNetID(1, 9))
    # exit(0)

    for id in AcasNetID.all_ids():
        _go(id)

    print('XL: Go manually check the outputs..')
    return
