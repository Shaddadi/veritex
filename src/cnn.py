import numpy as np
import copy as cp
from vzono import Vzono
import sys
import torch


class CNN:

    def __init__(self, sequential):
        self.sequential = sequential
        self.layer_num = len(sequential)
        self.images = None
        self.labels = None
        self._layer = None


    def reach_over_appr(self, vzono_set):

        for self._layer in range(self.layer_num):
            type_name = type(self.sequential[self._layer]).__name__
            # ['Conv2d', 'Linear', 'ReLU', 'Flatten']
            if type_name == 'Conv2d':
                vzono_set = self.conv2d(vzono_set)
            elif type_name == 'ReLU':
                vzono_set = self.relu_over_appr(vzono_set)
            elif type_name == 'Linear':
                vzono_set = self.linear(vzono_set)
            elif type_name == 'Flatten':
                vzono_set = self.flatten(vzono_set)
            else:
                sys.exit('This layer type is not supported yet!')

        return self.verify_vzono(vzono_set)


    def conv2d(self, vzono_set):
        base_vertices = vzono_set.base_vertices
        base_vectors = vzono_set.base_vectors
        self.sequential[self._layer]()

    def reluLayerLinearRelax(self, vzono_set):
        neurons_neg_pos, neurons_neg = self.get_valid_neurons_for_over_app(vzono_set)
        vzono_set.base_vertices[neurons_neg,:] = 0
        vzono_set.base_vectors[neurons_neg,:] = 0

        if neurons_neg_pos.shape[0] == 0:
            return vzono_set

        base_vectices = vzono_set.base_vertices[neurons_neg_pos,:]
        vals = np.sum(np.abs(vzono_set.base_vectors[neurons_neg_pos,:]), axis=1, keepdims=True)
        ubs = np.max(base_vectices,axis=1, keepdims=True) + vals
        lbs = np.min(base_vectices, axis=1, keepdims=True) - vals
        M = np.eye(vzono_set.base_vertices.shape[0])
        b = np.zeros((vzono_set.base_vertices.shape[0], 1))
        base_vectors_relax = np.zeros((vzono_set.base_vertices.shape[0],len(neurons_neg_pos)))

        A = ubs / (ubs - lbs)
        epsilons = -lbs*A/2
        M[neurons_neg_pos, neurons_neg_pos] = A[:,0]
        b[neurons_neg_pos] = epsilons
        base_vectors_relax[neurons_neg_pos, range(len(ubs))] = epsilons[:,0]

        new_base_vertices = np.dot(M,vzono_set.base_vertices) + b
        new_base_vectors = np.concatenate((np.dot(M, vzono_set.base_vectors), base_vectors_relax), axis=1)
        vzono_set.base_vertices = new_base_vertices
        vzono_set.base_vectors = new_base_vectors

        return vzono_set


    def get_valid_neurons_for_over_app(self, vfl_set):
        vals = np.sum(np.abs(vfl_set.base_vectors), axis=1, keepdims=True)
        temp_neg = np.all((vfl_set.base_vertices+vals) <= 0, 1)
        valid_neurons_neg = np.asarray(np.nonzero(temp_neg)).T[:, 0]
        temp_pos = np.all((vfl_set.base_vertices-vals) >= 0, 1)
        neurons_sum = temp_neg + temp_pos
        valid_neurons_neg_pos = np.asarray(np.nonzero(neurons_sum == False)).T[:, 0]

        return valid_neurons_neg_pos, valid_neurons_neg



    def singleLayerOverApp(self, vzono_set, layer_id):
        W = self._W[layer_id]
        b = self._b[layer_id]
        vzono_set.base_vertices = np.dot(W, vzono_set.base_vertices) + b
        vzono_set.base_vectors = np.dot(W, vzono_set.base_vectors)

        if layer_id == self._num_layer-1:
            return vzono_set

        over_app_set = self.reluLayerLinearRelax(vzono_set)

        return over_app_set


    def reachOverApp(self, state_tuple):
        vfl_set, layer, neurons = state_tuple # (vfl, layer, neurons)
        base_vertices = np.dot(vfl_set.M, vfl_set.vertices.T) + vfl_set.b
        base_vectors = np.zeros((base_vertices.shape[0], 1))
        vzono_set = Vzono(base_vertices, base_vectors)

        neurons_neg_pos, neurons_neg = self.get_valid_neurons_for_over_app(vzono_set)
        vzono_set.base_vertices[neurons_neg,:] = 0
        vzono_set.base_vectors[neurons_neg,:] = 0

        for n in range(layer+1, self._num_layer):
            vzono_set = self.singleLayerOverApp(vzono_set, n)

        return vzono_set


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





