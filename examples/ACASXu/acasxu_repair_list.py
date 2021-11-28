import torch

from acasxu_properties import *

# the list of neural networks that does not violate any of properties 1-10
# safe_nnet_list = [[1,1], [3,3], [4,2], [1,7], [1,8]]
safe_nnet_list = [[1,1],[1,2],[1,3],[1,4],[1,5],[1,6], [3,3], [4,2], [1,7], [1,8]]

# the list of properties that are violated by at least one neural network
# property7 is only for nnet19, property is only for nnet29
# violated_property = [property1, property2, property3, property4, property7, property8]

# correction function for each property where the closest safe data will be computed for each unsafe data
def repair_property1(unsafe_data):
    corrected_Ys = []
    original_Xs = []
    epsilon = 0.001
    for i in range(len(unsafe_data)):
        orig_x = torch.tensor(unsafe_data[i][0], dtype=torch.float32)
        unsafe_y = torch.tensor(unsafe_data[i][1], dtype=torch.float32)

        M, vec = property1.unsafe_domains[0], property1.unsafe_domains[1]
        res = torch.mm(M, unsafe_y.T) + vec
        delta_y = -res[0] + epsilon
        unsafe_y[0][0] = unsafe_y[0][0] + delta_y  # safe y
        corrected_Ys.append(unsafe_y)
        original_Xs.append(orig_x)

    corrected_Ys = torch.cat(corrected_Ys, dim=0)
    original_Xs = torch.cat(original_Xs, dim=0)
    return original_Xs, corrected_Ys


def repair_property2(unsafe_data):
    corrected_Ys = []
    original_Xs = []
    epsilon = 0.001
    for i in range(len(unsafe_data)):
        orig_x = torch.tensor(unsafe_data[i][0], dtype=torch.float32)
        unsafe_y = torch.tensor(unsafe_data[i][1], dtype=torch.float32)
        label = torch.argmax(unsafe_y * (-1))

        critical_dims = [0,label] # dim 0 for CoC
        M, vec = property2.unsafe_domains[0][0], property2.unsafe_domains[0][1]
        res = torch.mm(M, unsafe_y.T) + vec
        sorted, indices = torch.sort(res[:, 0], descending=True)
        for max_indx in indices:
            target_dim = torch.nonzero(M[max_indx] == 1)[0][0]
            if target_dim not in critical_dims:
                delta_y = (-res[max_indx, 0] + epsilon) / M[max_indx][target_dim]
                unsafe_y[0][target_dim] = unsafe_y[0][target_dim] + delta_y  # safe y
                break

        res = torch.mm(M, unsafe_y.T) + vec
        assert torch.any(res > 0, dim=0)
        corrected_Ys.append(unsafe_y)
        original_Xs.append(orig_x)

    corrected_Ys = torch.cat(corrected_Ys, dim=0)
    original_Xs = torch.cat(original_Xs, dim=0)
    return original_Xs, corrected_Ys



def repair_property3(unsafe_data):
    corrected_Ys = []
    original_Xs = []
    epsilon = 0.001
    for i in range(len(unsafe_data)):
        orig_x = torch.tensor(unsafe_data[i][0], dtype=torch.float32)
        unsafe_y = torch.tensor(unsafe_data[i][1], dtype=torch.float32)

        M, vec = property3.unsafe_domains[0], property3.unsafe_domains[1]
        res = torch.mm(M, unsafe_y.T) + vec
        max_indx = torch.argmax(res)
        target_dim = 0 # dim 0 for CoC
        delta_y = (-res[max_indx, 0] + epsilon) / M[max_indx][target_dim]
        unsafe_y[0][target_dim] = unsafe_y[0][target_dim] + delta_y  # safe y

        res = torch.mm(M, unsafe_y.T) + vec
        assert torch.any(res > 0, dim=0)
        corrected_Ys.append(unsafe_y)
        original_Xs.append(orig_x)

    corrected_Ys = torch.cat(corrected_Ys, dim=0)
    original_Xs = torch.cat(original_Xs, dim=0)
    return original_Xs, corrected_Ys


def repair_property4(unsafe_data):
    corrected_Ys = []
    original_Xs = []
    epsilon = 0.001
    for i in range(len(unsafe_data)):
        orig_x = torch.tensor(unsafe_data[i][0], dtype=torch.float32)
        unsafe_y = torch.tensor(unsafe_data[i][1], dtype=torch.float32)

        M, vec = property4.unsafe_domains[0], property4.unsafe_domains[1]
        res = torch.mm(M, unsafe_y.T) + vec
        max_indx = torch.argmax(res)
        target_dim = 0 # dim 0 for CoC
        delta_y = (-res[max_indx, 0] + epsilon) / M[max_indx][target_dim]
        unsafe_y[0][target_dim] = unsafe_y[0][target_dim] + delta_y  # safe y

        res = torch.mm(M, unsafe_y.T) + vec
        assert torch.any(res > 0, dim=0)
        corrected_Ys.append(unsafe_y)
        original_Xs.append(orig_x)

    corrected_Ys = torch.cat(corrected_Ys, dim=0)
    original_Xs = torch.cat(original_Xs, dim=0)
    return original_Xs, corrected_Ys


def repair_property7(unsafe_data):
    corrected_Ys = []
    original_Xs = []
    epsilon = 0.001
    for i in range(len(unsafe_data)):
        orig_x = torch.tensor(unsafe_data[i][0], dtype=torch.float32)
        unsafe_y = torch.tensor(unsafe_data[i][1], dtype=torch.float32)
        unsafe_domains = property7.unsafe_domains
        M0, vec0 = unsafe_domains[0][0], unsafe_domains[0][1]
        M1, vec1 = unsafe_domains[1][0], unsafe_domains[1][1]

        max0 = torch.max(torch.matmul(M0, unsafe_y.T)+vec0) # strong right
        max1 = torch.max(torch.matmul(M1, unsafe_y.T)+vec1) # strong left

        assert max0<=0.0 or max1<=0.0

        if (max1 >= 0) or (max0>=max1 and max0<=0.0): # strong right
            try:
                assert max0 <= 0
            except:
                xx = 1
            delta_y = -max0 + epsilon
            unsafe_y[0][3] = unsafe_y[0][3] + delta_y  # safe y
        elif (max0 >= 0) or (max1 >= max0 and max1<=0.0): # weak left
            assert max1 <= 0
            delta_y = -max1 + epsilon
            unsafe_y[0][4] = unsafe_y[0][4] + delta_y  # safe y

        res0 = torch.mm(M0, unsafe_y.T) + vec0
        res1 = torch.mm(M1, unsafe_y.T) + vec1
        assert torch.any(res0>0, dim=0) or torch.any(res1>0, dim=0)
        corrected_Ys.append(unsafe_y)
        original_Xs.append(orig_x)

    corrected_Ys = torch.cat(corrected_Ys, dim=0)
    original_Xs = torch.cat(original_Xs, dim=0)
    return original_Xs, corrected_Ys



def repair_property8(unsafe_data):
    corrected_Ys = []
    original_Xs = []
    epsilon = 0.001
    for i in range(len(unsafe_data)):
        orig_x = torch.tensor(unsafe_data[i][0], dtype=torch.float32)
        unsafe_y = torch.tensor(unsafe_data[i][1], dtype=torch.float32)

        M0, M1 = torch.cat(arry0)*(-1), torch.cat(arry1)*(-1)
        max0 = torch.max(torch.matmul(M0, unsafe_y.T)) # CoC
        max1 = torch.max(torch.matmul(M1, unsafe_y.T)) # weak left

        if max0 <= max1: # CoC
            assert max0 >= 0
            delta_y = -max0 - epsilon
            unsafe_y[0][0] = unsafe_y[0][0] + delta_y  # safe y
        else: # weak left
            assert max1 >= 0
            delta_y = -max1 - epsilon
            unsafe_y[0][1] = unsafe_y[0][1] + delta_y  # safe y

        res0 = torch.mm(M0, unsafe_y.T)
        res1 = torch.mm(M1, unsafe_y.T)
        assert torch.all(res0<0, dim=0) or torch.all(res1<0, dim=0)
        corrected_Ys.append(unsafe_y)
        original_Xs.append(orig_x)

    corrected_Ys = torch.cat(corrected_Ys, dim=0)
    original_Xs = torch.cat(original_Xs, dim=0)
    return original_Xs, corrected_Ys




# create neural networks that need to be repaired and their properties
repair_list = []
for i in range(1,6):
    for j in range(1,10):
        nnet = [i,j]
        if nnet in safe_nnet_list:
            continue
        property_ls =[[property1, repair_property1],
                      [property2, repair_property2],
                      [property3, repair_property3],
                      [property4, repair_property4]]
        if nnet == [1,9]:
            property_ls = [[property7, repair_property7]]
        elif nnet == [2,9]:
            property_ls.append([property8, repair_property8])

        repair_list.append([nnet, property_ls])











