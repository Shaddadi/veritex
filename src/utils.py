import numpy as np
import itertools
from vzono import VzonoFFNN as Vzono


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
    combs = (list(itertools.product(*V)))
    all_bounds = []
    for bound in combs:
        lbs = [item[0] for item in bound]
        ubs = [item[1] for item in bound]
        all_bounds.append([lbs, ubs])

    return all_bounds


