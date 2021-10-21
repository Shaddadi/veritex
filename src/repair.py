from ffnn import FFNN
import torch
import numpy as np
import copy as cp
import multiprocessing as mp
from worker import Worker
from shared import SharedState



class REPAIR:

    def __init__(self, properties, data, torch_model=None, weights=None, bias=None):
        self.properties = properties
        self.data = data
        if torch_model is not None:
            self.ffnn = self.construct_ffnn(torch_model)
        elif (weights is not None) and (bias is not None):
            self.ffnn = FFNN(weights, bias, unsafe_inputs=True)
        else:
            raise ValueError('Missing model parameters!')



    def construct_ffnn(self, torch_model):
        weights = []
        bias = []
        for name, param in torch_model.named_parameters():
            if name[-4:] == 'ight':
                if torch.cuda.is_available():
                    weights.append(param.data.cpu().numpy())
                else:
                    weights.append(param.data.numpy())
            if name[-4:] == 'bias':
                if torch.cuda.is_available():
                    temp = np.expand_dims(param.data.cpu().numpy(), axis=1)
                    bias.append(temp)
                else:
                    temp = np.expand_dims(param.data.numpy(), axis=1)
                    bias.append(temp)

        return FFNN(weights, bias, unsafe_inputs=True)



    def compute_unsafety(self):

        all_unsafe_data = []
        all_safe_data = []
        all_outputSets = []
        all_vfls_unsafe = []
        property_result = []
        self.ffnn.outputs_len = 100
        num_processors = mp.cpu_count()
        for n, prop in enumerate(self.properties):
            vfl_input = cp.deepcopy(prop.input_set)
            self.ffnn.unsafe_domains = prop.unsafe_domains
            processes = []
            unsafe_sets = []
            shared_state = SharedState([vfl_input], num_processors)
            one_worker = Worker(self.ffnn)
            for index in range(num_processors):
                p = mp.Process(target=one_worker.main_func, args=(index, shared_state))
                processes.append(p)
                p.start()

            for p in processes:
                p.join()

            while not shared_state.outputs.empty():
                unsafe_sets.append(shared_state.outputs.get())



    def correct_inputs(self, unsafe_data, epsilon=0.0001):
        corrected_ys = []
        original_Xs = []
        for n, aset in enumerate(unsafe_data):
            if len(aset[0]) == 0:
                continue

            X_unsafe = aset[0]
            Y_unsafe = aset[1]
            p = self.properties[n]
            unsafe_domains = p.unsafe_domains
            M, vec = torch.tensor(p[1][0]), torch.tensor(p[1][1])
            if torch.cuda.is_available():
                M, vec = M.cuda(), vec.cuda()

            for i in range(len(Y_unsafe)):
                unsafe_y = Y_unsafe[[i]]
                res = torch.matmul(M, unsafe_y.T) + vec
                if torch.any(res > 0):
                    continue

                min_indx = torch.argmax(res)

                delta_y = M[min_indx] * (-res[min_indx, 0] + epsilon) / (torch.matmul(M[[min_indx]], M[[min_indx]].T))
                safe_y = unsafe_y + delta_y
                corrected_ys.append(safe_y)
                original_Xs.append(X_unsafe[[i]])

        corrected_ys = torch.cat(corrected_ys, dim=0)
        original_Xs = torch.cat(original_Xs, dim=0)

        return original_Xs, corrected_ys




class DATA:
    def __init__(self, train_data, valid_data, test_data):
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data

