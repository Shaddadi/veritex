import numpy as np
import itertools
import sys
from vzono import VzonoFFNN as Vzono
import torch
from torch import nn
from diffabs.deeppoly import Dom


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def test_model():
    dom = Dom()
    absmodel = []
    relu = dom.ReLU()
    lin1 = dom.Linear(2, 2)  # from [x1, x2] to [x3, x4]
    lin2 = dom.Linear(2, 2)  # from [x5, x6] to [x7, x8]
    lin3 = dom.Linear(2, 2)  #
    with torch.no_grad():
        lin1.weight.data = torch.tensor([
            [1., 1.],
            [1., -1.]
        ])
        lin1.bias.data.zero_()

        lin2.weight.data = torch.tensor([
            [1., 1.],
            [1., -1.]
        ])
        lin2.bias.data.zero_()

        lin3.weight.data = torch.tensor([
            [1., 1.],
            [0., 1.]
        ])
        lin3.bias.data = torch.tensor([1., 0.])

    lin1 = lin1.to(device)
    lin2 = lin2.to(device)
    lin3 = lin3.to(device)
    absmodel.append(lin1)
    absmodel.append(relu)
    absmodel.append(lin2)
    absmodel.append(relu)
    absmodel.append(lin3)

    absmodel = nn.Sequential(*absmodel)
    return absmodel

def absmodel_from_torch(torch_model):
    dom = Dom()
    absmodel = []
    for layer in torch_model:
        if isinstance(layer, nn.Linear):
            w = layer.weight.data.shape
            liner = dom.Linear(w[1], w[0])
            liner.weight = layer.weight
            liner.bias = layer.bias
            liner.to(device)
            absmodel.append(liner)
        elif isinstance(layer, nn.ReLU):
            relu = dom.ReLU()
            relu.to(device)
            absmodel.append(relu)
        else:
            sys.exit('This type of layer is not supported yet')

    absmodel = nn.Sequential(*absmodel)
    return absmodel


def split_bounds_abs2(lbs, ubs):
    lbs = np.array(lbs)
    ubs = np.array(ubs)

    dim_split_times = [20,10,5,10,8]
    V = []
    for dim in range(len(lbs)):
        lb, ub = lbs[dim], ubs[dim]
        interval = (ub-lb)/(dim_split_times[dim])
        split_list = []
        for num in range(dim_split_times[dim]):
            split_list.append([lb+num*interval, lb+(num+1)*interval])

        V.append(split_list)

    combs = (list(itertools.product(*V)))
    all_bounds = []
    for bound in combs:
        lbs = [item[0] for item in bound]
        ubs = [item[1] for item in bound]
        all_bounds.append([lbs, ubs])

    all_lbs = torch.tensor([item[0] for item in all_bounds], device=device, dtype=torch.float32)
    all_ubs = torch.tensor([item[1] for item in all_bounds], device=device, dtype=torch.float32)
    dom = Dom()
    input = dom.Ele.by_intvl(all_lbs, all_ubs)
    return input



def split_bounds_abs(lbs, ubs, num=1):
    dom = Dom()
    all_bounds = [[lbs,ubs]]
    for _ in range(num):
        temp_list = []
        for bound in all_bounds:
            temp_list.extend(split_bounds_once_abs(bound[0], bound[1]))
        all_bounds = temp_list

    all_lbs = torch.tensor([item[0] for item in all_bounds], device=device)
    all_ubs = torch.tensor([item[1] for item in all_bounds], device=device)
    input = dom.Ele.by_intvl(all_lbs, all_ubs)
    return input


def split_bounds_once_abs(lbs, ubs):
    lbs = np.array(lbs)
    ubs = np.array(ubs)
    middle = (ubs + lbs)/2

    V = [[[lbs[n], middle[n]],[middle[n], ubs[n]]] for n in range(len(lbs))]
    combs = (list(itertools.product(*V)))
    all_bounds = []
    for bound in combs:
        lbs = [item[0] for item in bound]
        ubs = [item[1] for item in bound]
        all_bounds.append([lbs, ubs])

    return all_bounds



def split_bounds(lbs, ubs, num=1):
    all_bounds = [[lbs,ubs]]
    for _ in range(num):
        temp_list = []
        for bound in all_bounds:
            temp_list.extend(split_bounds_once(bound[0], bound[1]))
        all_bounds = temp_list

    sub_vzonos = []
    for item in all_bounds:
        lbs, ubs = item[0], item[1]
        base_vertices = np.array([(np.array(lbs)+np.array(ubs))/2]).T
        base_vectors = np.diag((np.array(ubs)-np.array(lbs))/2)
        sub_vzonos.append([Vzono(base_vertices, base_vectors),[lbs,ubs]])

    return sub_vzonos


def split_bounds_vset(lbs, ubs, num=1):
    all_bounds = [[lbs,ubs]]
    for _ in range(num):
        temp_list = []
        for bound in all_bounds:
            temp_list.extend(split_bounds_once(bound[0], bound[1]))
        all_bounds = temp_list

    sub_vsets = []
    for item in all_bounds:
        lbs, ubs = item[0], item[1]
        V = [[item[0][n], item[1][n]] for n in range(len(lbs))]
        vset = np.array(list(itertools.product(*V))).T
        sub_vsets.append([vset,[lbs, ubs]])

    return sub_vsets


def split_bounds_once(lbs, ubs):
    lbs = np.array(lbs)
    ubs = np.array(ubs)
    middle = (ubs + lbs)/2

    V = [[[lbs[n], middle[n]],[middle[n], ubs[n]]] for n in range(len(lbs))]
    # V = [[[lbs[n], middle[n]], [middle[n], ubs[n]]] for n in [0,1,2,3]]
    # n = 3
    # V.insert(n, [[lbs[n], ubs[n]]])
    # n = 4
    # V.insert(n, [[lbs[n], ubs[n]]])
    combs = (list(itertools.product(*V)))
    all_bounds = []
    for bound in combs:
        lbs = [item[0] for item in bound]
        ubs = [item[1] for item in bound]
        all_bounds.append([lbs, ubs])

    return all_bounds


