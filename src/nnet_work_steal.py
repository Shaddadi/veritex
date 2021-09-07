import numpy as np
import copy as cp
from vzono import Vzono
from itertools import product
from scipy.optimize import linprog
from collections import deque
import multiprocessing



class DNN:

    def __init__(self, W, b):
        self._W = W
        self._b = b
        self._num_layer = len(W)
        self.unsafe_domains = None

        # configurations for reachability analysis
        self.config_relu_linear = None
        self.config_only_verify = None
        self.config_unsafe_input = None
        self.config_exact_output = None


    def backtrack(self, vfl_set, verify=False):

        vfls = []
        for i in range(len(self.unsafe_domains)):
            As_unsafe = self.unsafe_domains[i][0]
            ds_unsafe = self.unsafe_domains[i][1]
            elements = np.dot(np.dot(As_unsafe,vfl_set.M), vfl_set.vertices.T) + np.dot(As_unsafe, vfl_set.b) +ds_unsafe
            if np.any(np.all(elements>0, axis=1)): # reachable set does not satisfy at least one linear constraint
                return None
            if np.any(np.all(elements<=0, axis=0)) and verify: # at least one vertex locates in unsafe domain
                return 0

            unsafe_vfl = cp.deepcopy(vfl_set)
            for j in range(len(As_unsafe)):
                A = As_unsafe[[j]]
                d = ds_unsafe[[j]]
                subvfl0 = unsafe_vfl.reluSplitHyperplane(A, d)
                if subvfl0:
                    unsafe_vfl = subvfl0
                else:
                    unsafe_vfl = None
                    return unsafe_vfl

            vfls.append(unsafe_vfl)

        return vfls


    def verify(self, vfl_set):
        vertices = np.dot(vfl_set.vertices, vfl_set.M.T) + vfl_set.b.T
        unsafe = False
        for ud in self.unsafe_domains:
            A_unsafe = ud[0]
            d_unsafe = ud[1]
            if len(A_unsafe) == 1:
                vals = np.dot(A_unsafe, vertices.T) + d_unsafe
                if np.any(np.all(vals<=0, axis=0)):
                    unsafe = True
                    break
            else:
                unsafe_vfl = self.backtrack(vfl_set, verify=True)
                if unsafe_vfl is not None:
                    unsafe = True
                    break

        return unsafe


    def compute_state(self, tuple_state):
        vfl_set, layer, neurons = tuple_state # (vfl, layer, neurons)

        if layer == self._num_layer - 1:  # the last layer
            return [(vfl_set, layer, None)]

        new_tuple_states = []
        if neurons.shape[0] == 0: # neurons empty, go to the next layer
            W = self._W[layer+1]
            b = self._b[layer+1]
            vfl_set.affineMap(W, b)
            new_tuple_states.append((vfl_set, layer+1, np.arange(vfl_set.M.shape[0])))
        else: # not empty
            new_vfl_sets, new_neurons = self.relu_neuron(vfl_set, neurons)
            for vfl in new_vfl_sets:
                new_tuple_states.append((vfl, layer, new_neurons))

        return new_tuple_states


    def relu_neuron(self, vfl_set, neurons):
        new_neurons = neurons
        if neurons.shape[0] == 0:
            return [vfl_set], new_neurons

        new_neurons, new_neurons_neg = self.get_valid_neurons(vfl_set, neurons)
        vfl_set.affineMapNegative(new_neurons_neg)

        if new_neurons.shape[0] == 0:
            return [vfl_set], new_neurons

        vfl_sets = self.relu_split(vfl_set, new_neurons[0])
        new_neurons = new_neurons[1:]
        return vfl_sets, new_neurons


    def relu_split(self, vfl_set, idx):
        outputPolySets = []
        sub_pos, sub_neg= vfl_set.reluSplit(idx)
        if sub_pos:
            outputPolySets.append(sub_pos)
        if sub_neg:
            outputPolySets.append(sub_neg)

        return outputPolySets




    def get_valid_neurons(self, vfl_set, neurons):
        assert neurons.shape[0]!=0

        elements = np.dot(vfl_set.vertices,vfl_set.M[neurons,:].T)+vfl_set.b[neurons,:].T
        flag_neg = (elements <= 0)
        temp_neg = np.all(flag_neg, 0)
        temp_pos = np.all(elements>=0, 0)
        temp_sum = temp_neg + temp_pos
        indx_neg_pos = np.asarray(np.nonzero(temp_sum == False)).T[:,0]
        valid_neurons_neg_pos = neurons[indx_neg_pos]
        indx_neg = np.asarray(np.nonzero(temp_neg)).T[:,0]
        valid_neurons_neg = neurons[indx_neg]

        return valid_neurons_neg_pos, valid_neurons_neg
