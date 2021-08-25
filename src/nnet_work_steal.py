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
