import os
import torch
from veritex.utils.vnnlib import vnnlib_to_properties

# get current directory
currdir = os.path.dirname(os.path.abspath(__file__))

# the list of neural networks that does not violate any of properties 1-10
safe_nnet_list = [[1,1],[1,2],[1,3],[1,4],[1,5],[1,6], [3,3], [4,2], [1,7], [1,8]]
hard_cases = [[1,9],[2,9]]


# normalize the whole input range to neural networks.
# this is for sampling of training data and test data when they are not available
ranges = torch.tensor([6.02610000e+04, 6.28318531e+00, 6.28318531e+00, 1.10000000e+03,
 1.20000000e+03, 3.73949920e+02])
means = torch.tensor([1.97910910e+04, 0.00000000e+00, 0.00000000e+00, 6.50000000e+02,
 6.00000000e+02, 7.51888402e+00])

#  [Clear-of-Conflict, weak left, weak right, strong left, strong right]
lbs_input = [0.0, -3.141593, -3.141593, 100.0, 0.0]
ubs_input = [60760.0, 3.141593, 3.141593, 1200.0, 1200.0]
for n in range(5):
    lbs_input[n] = (lbs_input[n] - means[n]) / ranges[n]
    ubs_input[n] = (ubs_input[n] - means[n]) / ranges[n]

input_ranges = [lbs_input, ubs_input]

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

        M, vec = property1.unsafe_domains[0][0], property1.unsafe_domains[0][1]
        M, vec = torch.tensor(M, dtype=torch.float32), torch.tensor(vec, dtype=torch.float32)
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
        M, vec = torch.tensor(M, dtype=torch.float32), torch.tensor(vec, dtype=torch.float32)
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

        M, vec = property3.unsafe_domains[0][0], property3.unsafe_domains[0][1]
        M, vec = torch.tensor(M, dtype=torch.float32), torch.tensor(vec, dtype=torch.float32)
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

        M, vec = property4.unsafe_domains[0][0], property4.unsafe_domains[0][1]
        M, vec = torch.tensor(M, dtype=torch.float32), torch.tensor(vec, dtype=torch.float32)
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
        M0, vec0 = torch.tensor(M0, dtype=torch.float32), torch.tensor(vec0, dtype=torch.float32)
        M1, vec1 = torch.tensor(M1, dtype=torch.float32), torch.tensor(vec1, dtype=torch.float32)

        max0 = torch.max(torch.matmul(M0, unsafe_y.T)+vec0) # strong right
        max1 = torch.max(torch.matmul(M1, unsafe_y.T)+vec1) # strong left

        assert max0<=0.0 or max1<=0.0

        if (max1 >= 0) or (max0>=max1 and max0<=0.0): # strong right
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

        # property 8 in vnncomp is slightly different from the original
        # in the repair, we consider the original property
        # x0>x1, x0>x2, x0>x3, x0>x4
        arry0 = [torch.tensor([[-1.0, 1.0, 0, 0, 0]]),
                 torch.tensor([[-1.0, 0, 1.0, 0, 0]]),
                 torch.tensor([[-1.0, 0, 0, 1.0, 0]]),
                 torch.tensor([[-1.0, 0, 0, 0, 1.0]]), ]
        # x1>x0, x1>x2, x1>x3, x1>x4
        arry1 = [torch.tensor([[1.0, -1.0, 0, 0, 0]]),
                 torch.tensor([[0, -1.0, 1.0, 0, 0]]),
                 torch.tensor([[0, -1.0, 0, 1.0, 0]]),
                 torch.tensor([[0, -1.0, 0, 0, 1.0]]),
                 ]
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

# extract properties from vnnlib
property1 = vnnlib_to_properties(f'{currdir}/../nets/prop_1.vnnlib', num_inputs=5, num_outputs=5, input_ranges=input_ranges)[0]
property2 = vnnlib_to_properties(f'{currdir}/../nets/prop_2.vnnlib', num_inputs=5, num_outputs=5, input_ranges=input_ranges)[0]
property3 = vnnlib_to_properties(f'{currdir}/../nets/prop_3.vnnlib', num_inputs=5, num_outputs=5, input_ranges=input_ranges)[0]
property4 = vnnlib_to_properties(f'{currdir}/../nets/prop_4.vnnlib', num_inputs=5, num_outputs=5, input_ranges=input_ranges)[0]
property7 = vnnlib_to_properties(f'{currdir}/../nets/prop_7.vnnlib', num_inputs=5, num_outputs=5, input_ranges=input_ranges)[0]
property8 = vnnlib_to_properties(f'{currdir}/../nets/prop_8.vnnlib', num_inputs=5, num_outputs=5, input_ranges=input_ranges)[0]

# create neural networks that need to be repaired and their properties
repair_list = []
for i in range(1,6):
    for j in range(1,10):
        nnet = [i,j]
        if nnet in safe_nnet_list or nnet in hard_cases:
            continue
        property_ls =[[property1, repair_property1],
                      [property2, repair_property2],
                      [property3, repair_property3],
                      [property4, repair_property4]]
        repair_list.append([nnet, property_ls])



# nnet [1,9], hard case
property_ls = [[property7, repair_property7]]
repair_list.append([[1,9],property_ls])

#nnet [2,9], hard case
property_ls= [[property1, repair_property1],
              [property2, repair_property2],
              [property3, repair_property3],
              [property4, repair_property4],
              [property8, repair_property8]]
repair_list.append([[2, 9], property_ls])









