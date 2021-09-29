import numpy as np
import copy as cp
from vzono import Vzono
import sys
import torch


class CNN:

    def __init__(self, sequential):
        self.sequential = sequential
        self.layer_num = len(sequential)
        self._layer = None


    def reach_over_appr(self, vzono_set, label):

        for self._layer in range(self.layer_num):
            type_name = type(self.sequential[self._layer]).__name__
            # ['Conv2d', 'Linear', 'ReLU', 'Flatten']
            if type_name == 'Conv2d':
                vzono_set = self.conv2d(vzono_set)
            elif type_name == 'ReLU':
                if len(vzono_set.base_vectors.shape) == 4:
                    vzono_set = self.relu_over_appr(vzono_set)
                elif len(vzono_set.base_vectors.shape) == 2: # fully connected layers
                    vzono_set = self.relu_over_appr2(vzono_set)
            elif type_name == 'Linear':
                vzono_set = self.linear(vzono_set)
            elif type_name == 'Flatten':
                vzono_set = self.flatten(vzono_set)
            else:
                sys.exit('This layer type is not supported yet!')

        return self.verify_vzono(vzono_set, label)


    def verify_vzono(self, vzono_set, label):
        vals = torch.sum(torch.abs(vzono_set.base_vectors), dim=0)
        ubs = vzono_set.base_vertices + vals
        lbs = vzono_set.base_vertices - vals

        if len(torch.nonzero(ubs >= lbs[label])) >= 2:
            return False # unsafe
        else:
            return True # safe


    def conv2d(self, vzono_set):
        vzono_set.base_vertices = self.sequential[self._layer](vzono_set.base_vertices)
        bias = self.sequential[self._layer].bias.data
        bias = torch.reshape(bias,(1,len(bias),1,1))
        vzono_set.base_vectors = self.sequential[self._layer](vzono_set.base_vectors) - bias

        return vzono_set


    def linear(self, vzono_set):
        vzono_set.base_vertices = self.sequential[self._layer](vzono_set.base_vertices)
        bias = self.sequential[self._layer].bias.data
        bias = torch.reshape(bias, (1, len(bias)))
        vzono_set.base_vectors = self.sequential[self._layer](vzono_set.base_vectors) - bias
        return vzono_set


    def flatten(self, vzono_set):
        vzono_set.base_vertices = torch.flatten(vzono_set.base_vertices)
        vzono_set.base_vectors = torch.reshape(vzono_set.base_vectors, (vzono_set.base_vectors.shape[0],-1))
        vzono_set.base_vectors = vzono_set.base_vectors.to_sparse()
        return  vzono_set


    def relu_over_appr(self, vzono_set):
        neurons_neg_pos, neurons_neg, vals = self.get_valid_neurons_for_over_app(vzono_set)
        vzono_set.base_vertices[:, neurons_neg] = 0
        vzono_set.base_vectors[:, neurons_neg] = 0

        if not torch.any(neurons_neg_pos):
            return vzono_set

        # base_vectices = vzono_set.base_vertices[:, neurons_neg_pos]
        base_vectices_max, _ = torch.max(vzono_set.base_vertices[:, neurons_neg_pos], dim=0, keepdim=True)
        base_vectices_min, _ = torch.min(vzono_set.base_vertices[:, neurons_neg_pos], dim=0, keepdim=True)

        ubs = base_vectices_max + vals[:, neurons_neg_pos]
        lbs = base_vectices_min - vals[:, neurons_neg_pos]
        M = ubs / (ubs - lbs)
        epsilons = -lbs * M / 2

        vzono_set.base_vertices[:, neurons_neg_pos] = vzono_set.base_vertices[:, neurons_neg_pos] * M + epsilons
        vzono_set.base_vectors[:, neurons_neg_pos] = vzono_set.base_vectors[:, neurons_neg_pos] * M

        neurons_neg_pos_index = torch.nonzero(neurons_neg_pos)
        new_base_vectors_shape = [neurons_neg_pos_index.shape[0]]
        new_base_vectors_shape.extend(list(neurons_neg_pos.size()))
        new_base_vectors = torch.zeros(tuple(new_base_vectors_shape))
        n = 0
        for index in neurons_neg_pos_index:
            new_base_vectors[n, index[0],index[1],index[2]] = epsilons[0,n]
            n += 1

        vzono_set.base_vectors = torch.cat((vzono_set.base_vectors, new_base_vectors), dim=0)
        return vzono_set


    def relu_over_appr2(self, vzono_set):
        neurons_neg_pos, neurons_neg, vals = self.get_valid_neurons_for_over_app2(vzono_set)
        vzono_set.base_vertices[neurons_neg] = 0
        vzono_set.base_vectors[:,neurons_neg] = 0

        if not torch.any(neurons_neg_pos):
            return vzono_set

        # base_vectices = vzono_set.base_vertices[:, neurons_neg_pos]
        base_vectices_max, _ = torch.max(vzono_set.base_vertices[neurons_neg_pos], dim=0, keepdim=True)
        base_vectices_min, _ = torch.min(vzono_set.base_vertices[neurons_neg_pos], dim=0, keepdim=True)

        ubs = base_vectices_max + vals[:, neurons_neg_pos]
        lbs = base_vectices_min - vals[:, neurons_neg_pos]
        M = ubs / (ubs - lbs)
        epsilons = -lbs * M / 2

        vzono_set.base_vertices[neurons_neg_pos] = vzono_set.base_vertices[neurons_neg_pos] * M + epsilons
        vzono_set.base_vectors[:, neurons_neg_pos] = vzono_set.base_vectors[:, neurons_neg_pos] * M

        neurons_neg_pos_index = torch.nonzero(neurons_neg_pos)
        new_base_vectors_shape = [len(neurons_neg_pos_index), neurons_neg_pos.shape[0]]
        new_base_vectors = torch.zeros(tuple(new_base_vectors_shape))
        new_base_vectors[range(new_base_vectors_shape[0]), neurons_neg_pos_index[:,0]] = epsilons

        vzono_set.base_vectors = torch.cat((vzono_set.base_vectors, new_base_vectors), dim=0)
        return vzono_set



    def get_valid_neurons_for_over_app(self, vfl_set):
        vals = torch.sum(torch.abs(vfl_set.base_vectors), dim=0, keepdim=True)
        valid_neurons_neg = torch.all((vfl_set.base_vertices+vals)<=0, dim=0)
        valid_neurons_pos = torch.all((vfl_set.base_vertices-vals)>=0, dim=0)
        valid_neurons_neg_pos = ~(valid_neurons_neg + valid_neurons_pos)

        return valid_neurons_neg_pos, valid_neurons_neg, vals


    def get_valid_neurons_for_over_app2(self, vfl_set):
        vals = torch.sum(torch.abs(vfl_set.base_vectors), dim=0, keepdim=True)
        valid_neurons_neg = torch.all((vfl_set.base_vertices+vals)<=0, dim=0)
        valid_neurons_pos = torch.all((vfl_set.base_vertices-vals)>=0, dim=0)
        valid_neurons_neg_pos = ~(valid_neurons_neg + valid_neurons_pos)

        return valid_neurons_neg_pos, valid_neurons_neg, vals







    def verifyVzono(self, vzono_set):
        safe = True
        for ud in self.unsafe_domains:
            As_unsafe = ud[0]
            ds_unsafe = ud[1]
            for n in range(len(As_unsafe)):
                A = As_unsafe[[n]]
                d = ds_unsafe[[n]]
                base_vertices = np.dot(A, vzono_set.base_vertices) + d
                base_vectors = np.dot(A, vzono_set.base_vectors)
                vals = base_vertices - np.sum(np.abs(base_vectors),axis=1)
                if np.any(vals<=0):
                    safe = False
                    return safe

        return safe





