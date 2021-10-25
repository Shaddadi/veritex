import numpy as np
import copy as cp
from vzono import VzonoFFNN as Vzono
import torch
from itertools import product
from scipy.optimize import linprog
from collections import deque
import multiprocessing



class FFNN:

    def __init__(self, torch_model, verify=False, relu_linear=False, unsafe_inputs=False, exact_output=False, repair=False):
        self.torch_model = torch_model
        self.extract_weights()
        self.unsafe_domains = None

        # configurations for reachability analysis
        self.config_verify = verify
        self.config_relu_linear = relu_linear
        self.config_unsafe_input = unsafe_inputs
        self.config_exact_output = exact_output
        self.config_repair = repair

        # relu linearization does not support computation of unsafe input domains and exact output domains
        assert not(self.config_exact_output and self.config_relu_linear)
        assert not (self.config_unsafe_input and self.config_relu_linear)


    def extract_weights(self):
        self._W = []
        self._b = []
        for name, param in self.torch_model.named_parameters():
            if name[-4:] == 'ight':
                if torch.cuda.is_available():
                    self._W.append(param.data.cpu().numpy())
                else:
                    self._W.append(param.data.numpy())
            if name[-4:] == 'bias':
                if torch.cuda.is_available():
                    temp = np.expand_dims(param.data.cpu().numpy(), axis=1)
                    self._b.append(temp)
                else:
                    temp = np.expand_dims(param.data.numpy(), axis=1)
                    self._b.append(temp)

        self._num_layer = len(self._W)



    def backtrack(self, vfl_set, verify=False, unsafe_domain=None):

        if verify:
            As_unsafe = unsafe_domain[0]
            ds_unsafe = unsafe_domain[1]
            elements = np.dot(np.dot(As_unsafe, vfl_set.M), vfl_set.vertices.T) + np.dot(As_unsafe, vfl_set.b) + ds_unsafe
            if np.any(np.all(elements >= 0, axis=1)):  # reachable set does not satisfy at least one linear constraint
                return False
            if np.any(np.all(elements <= 0, axis=0)):  # at least one vertex locates in unsafe domain
                return True
            unsafe_vfl = cp.deepcopy(vfl_set)
            for j in range(len(As_unsafe)):
                A = As_unsafe[[j]]
                d = ds_unsafe[[j]]
                subvfl0 = unsafe_vfl.reluSplitHyperplane(A, d)
                if subvfl0:
                    unsafe_vfl = subvfl0
                else:
                    return False # vfl_set does not contain any unsafe elements
            return True # unsafe_vfl is not none and contains unsafe elements
        else:
            vfls = []
            for i in range(len(self.unsafe_domains)):
                As_unsafe = self.unsafe_domains[i][0].numpy()
                ds_unsafe = self.unsafe_domains[i][1].numpy()
                elements = np.dot(np.dot(As_unsafe,vfl_set.M), vfl_set.vertices.T) + np.dot(As_unsafe, vfl_set.b) +ds_unsafe
                if np.any(np.all(elements>0, axis=1)): # reachable set does not satisfy at least one linear constraint
                    continue

                unsafe_vfl = cp.deepcopy(vfl_set)
                for j in range(len(As_unsafe)):
                    A = As_unsafe[[j]]
                    d = ds_unsafe[[j]]
                    subvfl0 = unsafe_vfl.reluSplitHyperplane(A, d)
                    if subvfl0:
                        unsafe_vfl = subvfl0
                    else:
                        unsafe_vfl = []
                        break

                if unsafe_vfl:
                    vfls.append(unsafe_vfl)

            return vfls



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


    def verify(self, vfl_set):
        unsafe = False
        for ud in self.unsafe_domains:
            A_unsafe = ud[0]
            d_unsafe = ud[1]
            if len(A_unsafe) == 1:
                vertices = np.dot(vfl_set.vertices, vfl_set.M.T) + vfl_set.b.T
                vals = np.dot(A_unsafe, vertices.T) + d_unsafe
                if np.any(np.all(vals<=0, axis=0)):
                    unsafe = True
                    break
            else:
                unsafe = self.backtrack(vfl_set, verify=True, unsafe_domain=ud)
                if unsafe:
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
