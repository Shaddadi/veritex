
from agent_properties import *
import torch


def repair_property1(unsafe_data):
    corrected_Ys = []
    original_Xs = []
    epsilon = 0.1
    for i in range(len(unsafe_data)):
        orig_x = torch.tensor(unsafe_data[i][0], dtype=torch.float32)
        unsafe_y = torch.tensor(unsafe_data[i][1], dtype=torch.float32)

        M, vec = property1.unsafe_domains[0][0], property1.unsafe_domains[0][1]
        M, vec = torch.tensor(M, dtype=torch.float32), torch.tensor(vec, dtype=torch.float32)
        res = torch.mm(M, unsafe_y.T) + vec
        values = torch.max(res[0], dim=0).values - epsilon
        indices = torch.max(res[0], dim=0).indices
        delta_y = torch.tensor([0.0,0.0,0.0])
        delta_y[indices+1] = values
        unsafe_y[0] = unsafe_y[0] + delta_y  # safe y
        corrected_Ys.append(unsafe_y)
        original_Xs.append(orig_x)

    corrected_Ys = torch.cat(corrected_Ys, dim=0)
    original_Xs = torch.cat(original_Xs, dim=0)
    return original_Xs, corrected_Ys


def repair_property2(unsafe_data):
    corrected_Ys = []
    original_Xs = []
    epsilon = 0.1
    for i in range(len(unsafe_data)):
        orig_x = torch.tensor(unsafe_data[i][0], dtype=torch.float32)
        unsafe_y = torch.tensor(unsafe_data[i][1], dtype=torch.float32)

        M, vec = property2.unsafe_domains[0][0], property2.unsafe_domains[0][1]
        M, vec = torch.tensor(M, dtype=torch.float32), torch.tensor(vec, dtype=torch.float32)
        res = torch.mm(M, unsafe_y.T) + vec
        values = torch.min(-res[0], dim=0).values + epsilon
        indices = torch.min(-res[0], dim=0).indices
        delta_y = torch.tensor([0.0, 0.0, 0.0])
        delta_y[indices + 1] = values
        unsafe_y[0] = unsafe_y[0] + delta_y  # safe y
        corrected_Ys.append(unsafe_y)
        original_Xs.append(orig_x)

    corrected_Ys = torch.cat(corrected_Ys, dim=0)
    original_Xs = torch.cat(original_Xs, dim=0)
    return original_Xs, corrected_Ys


# create neural networks that need to be repaired and their properties
repair_list = []
repair_list.append([0, [[property1, repair_property1],[property2, repair_property2]]])
repair_list.append([1, [[property1, repair_property1],[property2, repair_property2]]])
repair_list.append([2, [[property1, repair_property1],[property2, repair_property2]]])













