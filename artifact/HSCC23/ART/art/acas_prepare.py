""" Since the training dataset for ACAS Xu is not publicly available,
    we uniformly generate and inspect the training/test set here.
"""

import sys
from pathlib import Path
from typing import List

import torch

from diffabs import AbsDom, DeeppolyDom

from art.prop import AndProp

sys.path.append(str(Path(__file__).resolve().parent.parent))

from art.acas import ACAS_DIR, AcasNetID, AcasNet, AcasOut
from art.utils import sample_points


def sample_original_data(dom: AbsDom, trainsize: int = 10000, testsize: int = 5000, dir: str = ACAS_DIR):
    """ Sample the data from every trained network. Serve as training and test set.
    :param dom: the data preparation do not use abstraction domains, although the AcasNet constructor requires it.
    """
    for nid in AcasNetID.all_ids():
        fpath = nid.fpath()
        print('\rSampling for network', nid, 'picked nnet file:', fpath, end='')
        net, bound_mins, bound_maxs = AcasNet.load_nnet(fpath, dom)
        net = net.to(device)

        in_lbs = torch.tensor([bound_mins], device=device)
        in_ubs = torch.tensor([bound_maxs], device=device)

        in_lbs = net.normalize_inputs(in_lbs, bound_mins, bound_maxs)
        in_ubs = net.normalize_inputs(in_ubs, bound_mins, bound_maxs)
        inputs = sample_points(in_lbs, in_ubs, K=trainsize+testsize)

        with torch.no_grad():
            outputs = net(inputs)
            labels = (outputs * -1).argmax(dim=-1)  # because in ACAS Xu, minimum score is the prediction

        # # it seems the prediction scores from original ACAS Xu network is very close
        # softmax = torch.nn.Softmax(dim=1)
        # loss = torch.nn.CrossEntropyLoss()
        # print(loss(softmax(outputs * -1), labels))

        train_inputs, test_inputs = inputs[:trainsize, ...], inputs[trainsize:, ...]
        train_labels, test_labels = labels[:trainsize, ...], labels[trainsize:, ...]

        torch.save((train_inputs, train_labels), Path(dir, f'{str(nid)}-orig-train.pt'))
        torch.save((test_inputs, test_labels), Path(dir, f'{str(nid)}-orig-test.pt'))
        print('\rSampled for network', nid, 'picked nnet file:', fpath)
    return


def sample_balanced_data(dom: AbsDom, trainsize: int = 10000, testsize: int = 5000, dir: str = ACAS_DIR):
    """ Sample the data from every trained network. Serve as training and test set.
        Note that the default original dataset is very imbalanced, we instead sample a balanced dataset
        where every category has exactly the same amount of data.
        Note that this applies to N_{1,1} ~ N_{1,6} only. Other networks all lack of data for certain categories.
        Some categories are having data < 0.1% of all sampled points.
    """
    assert trainsize % len(AcasOut) == 0 and testsize % len(AcasOut) == 0

    for nid in AcasNetID.balanced_ids():
        fpath = nid.fpath()
        print('Sampling for network', nid, 'picked nnet file:', fpath)
        net, bound_mins, bound_maxs = AcasNet.load_nnet(fpath, dom)
        net = net.to(device)

        in_lbs = torch.tensor([bound_mins], device=device)
        in_ubs = torch.tensor([bound_maxs], device=device)

        in_lbs = net.normalize_inputs(in_lbs, bound_mins, bound_maxs)
        in_ubs = net.normalize_inputs(in_ubs, bound_mins, bound_maxs)

        res_inputs = [torch.tensor([]) for _ in range(len(AcasOut))]
        res_labels = [torch.tensor([]).long() for _ in range(len(AcasOut))]

        trainsize_cat = int(trainsize / len(AcasOut))
        testsize_cat = int(testsize / len(AcasOut))
        allsize_cat = trainsize_cat + testsize_cat
        while True:
            inputs = sample_points(in_lbs, in_ubs, K=trainsize+testsize)
            with torch.no_grad():
                outputs = net(inputs)
                labels = (outputs * -1).argmax(dim=-1)  # because in ACAS Xu, minimum score is the prediction

            all_filled = True
            for category in AcasOut:
                if len(res_inputs[category]) >= allsize_cat:
                    continue

                all_filled = False
                idxs = labels == category
                cat_inputs, cat_labels = inputs[idxs], labels[idxs]
                res_inputs[category] = torch.cat((res_inputs[category], cat_inputs), dim=0)
                res_labels[category] = torch.cat((res_labels[category], cat_labels), dim=0)

            if all_filled:
                break
            pass

        empty = torch.tensor([])
        train_inputs, train_labels = empty, empty.long()
        test_inputs, test_labels = empty, empty.long()

        for category in AcasOut:
            cat_inputs, cat_labels = res_inputs[category], res_labels[category]
            train_inputs = torch.cat((train_inputs, cat_inputs[:trainsize_cat, ...]), dim=0)
            train_labels = torch.cat((train_labels, cat_labels[:trainsize_cat, ...]), dim=0)
            test_inputs = torch.cat((test_inputs, cat_inputs[trainsize_cat:trainsize_cat+testsize_cat, ...]), dim=0)
            test_labels = torch.cat((test_labels, cat_labels[trainsize_cat:trainsize_cat+testsize_cat, ...]), dim=0)
            pass

        # # it seems the prediction scores from original ACAS Xu network is very close
        # softmax = torch.nn.Softmax(dim=1)
        # loss = torch.nn.CrossEntropyLoss()
        # print(loss(softmax(outputs * -1), labels))

        with open(Path(dir, f'{str(nid)}-normed-train.pt'), 'wb') as f:
            torch.save((train_inputs, train_labels), f)
        with open(Path(dir, f'{str(nid)}-normed-test.pt'), 'wb') as f:
            torch.save((test_inputs, test_labels), f)
    return


def sample_balanced_data_for(dom: AbsDom, nid: AcasNetID, ignore_idxs: List[int],
                             trainsize: int = 10000, testsize: int = 5000, dir: str = ACAS_DIR):
    """ Some networks' original data is soooooo imbalanced.. Some categories are ignored. """
    assert len(ignore_idxs) != 0, 'Go to the other function.'
    assert all([0 <= i < len(AcasOut) for i in ignore_idxs])
    print('Sampling for', nid, 'ignoring output category', ignore_idxs)

    ncats = len(AcasOut) - len(ignore_idxs)
    train_percat = int(trainsize / ncats)
    test_percat = int(testsize / ncats)

    def trainsize_of(i: AcasOut):
        return 0 if i in ignore_idxs else train_percat

    def testsize_of(i: AcasOut):
        return 0 if i in ignore_idxs else test_percat

    fpath = nid.fpath()
    print('Sampling for network', nid, 'picked nnet file:', fpath)
    net, bound_mins, bound_maxs = AcasNet.load_nnet(fpath, dom)
    net = net.to(device)

    in_lbs = torch.tensor([bound_mins], device=device)
    in_ubs = torch.tensor([bound_maxs], device=device)

    in_lbs = net.normalize_inputs(in_lbs, bound_mins, bound_maxs)
    in_ubs = net.normalize_inputs(in_ubs, bound_mins, bound_maxs)

    res_inputs = [torch.tensor([]) for _ in range(len(AcasOut))]
    res_labels = [torch.tensor([]).long() for _ in range(len(AcasOut))]

    while True:
        inputs = sample_points(in_lbs, in_ubs, K=trainsize + testsize)
        with torch.no_grad():
            outputs = net(inputs)
            labels = (outputs * -1).argmax(dim=-1)  # because in ACAS Xu, minimum score is the prediction

        filled_cnt = 0
        for category in AcasOut:
            if len(res_inputs[category]) >= trainsize_of(category) + testsize_of(category):
                filled_cnt += 1

            if category not in ignore_idxs and len(res_inputs[category]) >= trainsize_of(category) + testsize_of(category):
                continue

            idxs = labels == category
            cat_inputs, cat_labels = inputs[idxs], labels[idxs]

            res_inputs[category] = torch.cat((res_inputs[category], cat_inputs), dim=0)
            res_labels[category] = torch.cat((res_labels[category], cat_labels), dim=0)
            pass

        if filled_cnt == len(AcasOut):
            break
        pass

    empty = torch.tensor([])
    train_inputs, train_labels = empty, empty.long()
    test_inputs, test_labels = empty, empty.long()

    for category in AcasOut:
        cat_inputs, cat_labels = res_inputs[category], res_labels[category]
        if category in ignore_idxs:
            amount = len(cat_inputs)
            pivot = int(amount * trainsize / (trainsize + testsize))
            train_inputs = torch.cat((train_inputs, cat_inputs[:pivot, ...]), dim=0)
            train_labels = torch.cat((train_labels, cat_labels[:pivot, ...]), dim=0)
            test_inputs = torch.cat((test_inputs, cat_inputs[pivot:, ...]), dim=0)
            test_labels = torch.cat((test_labels, cat_labels[pivot:, ...]), dim=0)
        else:
            trainsize_cat = trainsize_of(category)
            testsize_cat = testsize_of(category)
            train_inputs = torch.cat((train_inputs, cat_inputs[:trainsize_cat, ...]), dim=0)
            train_labels = torch.cat((train_labels, cat_labels[:trainsize_cat, ...]), dim=0)
            test_inputs = torch.cat((test_inputs, cat_inputs[trainsize_cat:trainsize_cat + testsize_cat, ...]), dim=0)
            test_labels = torch.cat((test_labels, cat_labels[trainsize_cat:trainsize_cat + testsize_cat, ...]), dim=0)
        pass

    # # it seems the prediction scores from original ACAS Xu network is very close
    # softmax = torch.nn.Softmax(dim=1)
    # loss = torch.nn.CrossEntropyLoss()
    # print(loss(softmax(outputs * -1), labels))

    with open(Path(dir, f'{str(nid)}-normed-train.pt'), 'wb') as f:
        torch.save((train_inputs, train_labels), f)
    with open(Path(dir, f'{str(nid)}-normed-test.pt'), 'wb') as f:
        torch.save((test_inputs, test_labels), f)
    return


def inspect_data_for(dom: AbsDom, nid: AcasNetID, dir: str = ACAS_DIR, normed: bool = True):
    """ Inspect the sampled data from every trained network. To serve as training and test set. """
    fpath = nid.fpath()
    print('Loading sampled data for network', nid, 'picked nnet file:', fpath)
    props = AndProp(nid.applicable_props(dom))
    print('Shall satisfy', props.name)
    net, bound_mins, bound_maxs = AcasNet.load_nnet(fpath, dom)
    net = net.to(device)

    mid = 'normed' if normed else 'orig'
    train_inputs, train_labels = torch.load(Path(dir, f'{str(nid)}-{mid}-train.pt'), device)
    test_inputs, test_labels = torch.load(Path(dir, f'{str(nid)}-{mid}-test.pt'), device)

    assert len(train_inputs) == len(train_labels)
    assert len(test_inputs) == len(test_labels)
    print(f'Loaded {len(train_inputs)} training samples, {len(test_inputs)} test samples.')

    for category in AcasOut:
        cnt = (train_labels == category).sum().item() + (test_labels == category).sum().item()
        print(f'Category {category} has {cnt} samples.')
    print()

    with torch.no_grad():
        # because in ACAS Xu, minimum score is the prediction
        assert torch.equal(train_labels, (net(train_inputs) * -1).argmax(dim=-1))
        assert torch.equal(test_labels, (net(test_inputs) * -1).argmax(dim=-1))
    return


if __name__ == '__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    dom = DeeppolyDom()

    sample_original_data(dom)
    # sample_balanced_data(dom)

    # # Ignore idxs are inspected from load_sampled_data(). Network <1, 1> and <1, 2> are already done.
    # sample_balanced_data_for(dom, AcasNetID(1, 7), [3, 4])
    # sample_balanced_data_for(dom, AcasNetID(1, 9), [1, 2, 3, 4])
    # sample_balanced_data_for(dom, AcasNetID(2, 1), [2])
    # sample_balanced_data_for(dom, AcasNetID(2, 9), [2, 3, 4])
    # sample_balanced_data_for(dom, AcasNetID(3, 3), [1])
    # sample_balanced_data_for(dom, AcasNetID(4, 5), [2, 4])

    print(len(AcasNetID.all_ids()))
    for nid in AcasNetID.all_ids():
        inspect_data_for(dom, nid, normed=False)
    print('All prepared ACAS dataset loaded and validated.')
    pass
