import numpy as np
import copy as cp
from veritex.sets.vzono import VzonoFFNN as Vzono
from veritex.networks.funcs import relu
from veritex.networks.funcs import tanh
from veritex.networks.funcs import sigmoid
import torch.nn as nn
import torch




class FFNN:
    """
    A class to construct a network model and conduct reachability analysis

    ...
    Attributes
    ----------
    _W: list
        weight matrix between the network layers
    _b: list
        bias vector between the network layers
    _f: list
        activation functions in each layer

    Methods
    _______
    extract_params(torch_model):
        extract parameters such as weights, bias and activation functions from a torch model
    """
    def __init__(self, model,
                 repair=False,
                 verification=False,
                 linearization=False,
                 unsafe_inputd=False,
                 exact_outputd=False):

        if isinstance(model, torch.nn.Sequential):
            self.extract_params(model)
        else:
            self._W, self._b, self._f = model

        assert len(self._f)==len(self._W) or len(self._f)+1==len(self._W)
        assert len(self._W)==len(self._b)
        self._num_layer = len(self._W)

        self.unsafe_domains = None

        # configurations for reachability analysis
        self.verification = verification
        self.linearization = linearization # relu linearization for over approximation
        self.unsafe_inputd = unsafe_inputd
        self.exact_outputd = exact_outputd
        self.repair = repair

        # relu linearization does not support computation of unsafe input domains and exact output domains
        assert not(self.exact_outputd and self.linearization)
        assert not (self.unsafe_inputd and self.linearization)


    def extract_params(self, torch_model):
        self._W = []
        self._b = []
        for name, param in torch_model.named_parameters():
            if name[-4:] == 'ight':
                if torch.cuda.is_available():
                    self._W.append(cp.deepcopy(param.data.cpu().numpy()))
                else:
                    self._W.append(cp.deepcopy(param.data.numpy()))
            if name[-4:] == 'bias':
                if torch.cuda.is_available():
                    temp = np.expand_dims(cp.deepcopy(param.data.cpu().numpy()), axis=1)
                    self._b.append(temp)
                else:
                    temp = np.expand_dims(cp.deepcopy(param.data.numpy()), axis=1)
                    self._b.append(temp)

        self._f = []
        for layer in torch_model:
            if isinstance(layer, nn.ReLU):
                self._f.append('ReLU')
            elif isinstance(layer, nn.Sigmoid):
                self._f.append('Sigmoid')
            elif isinstance(layer, nn.Tanh):
                self._f.append('Tanh')


    def backtrack(self, s, verify=False, unsafe_domain=None):
        if verify:
            As_unsafe = unsafe_domain[0]
            ds_unsafe = unsafe_domain[1]
            elements = np.dot(np.dot(As_unsafe, s.M), s.vertices.T) + np.dot(As_unsafe, s.b) + ds_unsafe
            if np.any(np.all(elements >= 0, axis=1)):  # reachable set does not satisfy at least one linear constraint
                return False
            if np.any(np.all(elements <= 0, axis=0)):  # at least one vertex locates in unsafe domain
                return True
            unsafe_s = cp.deepcopy(s)
            for j in range(len(As_unsafe)):
                A = As_unsafe[[j]]
                d = ds_unsafe[[j]]
                sub0 = unsafe_s.reluSplitHyperplane(A, d)
                if sub0:
                    unsafe_s = sub0
                else:
                    return False # vfl_set does not contain any unsafe elements
            return True # unsafe_vfl is not none and contains unsafe elements
        else:
            vfls = []
            for i in range(len(self.unsafe_domains)):
                As_unsafe = self.unsafe_domains[i][0]
                ds_unsafe = self.unsafe_domains[i][1]
                elements = np.dot(np.dot(As_unsafe,s.M), s.vertices.T) + np.dot(As_unsafe, s.b) +ds_unsafe
                if np.any(np.all(elements>0, axis=1)): # reachable set does not satisfy at least one linear constraint
                    continue

                unsafe_vfl = cp.deepcopy(s)
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



    def layer_over_approximation(self, s, l):
        W = self._W[l]
        b = self._b[l]
        s.base_vertices = np.dot(W, s.base_vertices) + b
        s.base_vectors = np.dot(W, s.base_vectors)
        if l == self._num_layer-1:
            return s

        over_app_set = relu.layer_linearize(s)
        return over_app_set


    def reach_over_app_nontuple(self, s):
        for n in range(self._num_layer):
            s = self.layer_over_approximation(s, n)
        return s


    def reach_over_app(self, state_tuple):
        s, layer, neurons = state_tuple # (vfl, layer, neurons)
        base_vertices = np.dot(s.M, s.vertices.T) + s.b
        base_vectors = np.zeros((base_vertices.shape[0], 1))
        vzono_set = Vzono(base_vertices, base_vectors)

        neurons_neg_pos, neurons_neg = vzono_set.get_valid_neurons_for_over_app()
        vzono_set.base_vertices[neurons_neg,:] = 0
        vzono_set.base_vectors[neurons_neg,:] = 0

        for n in range(layer+1, self._num_layer):
            vzono_set = self.layer_over_approximation(vzono_set, n)

        return vzono_set


    def verify_vzono(self, s):

        safe = []
        for indx, ud in enumerate(self.unsafe_domains):
            As_unsafe = ud[0]
            ds_unsafe = ud[1]
            safe.append(False)
            for n in range(len(As_unsafe)):
                A = As_unsafe[[n]]
                d = ds_unsafe[[n]]
                base_vertices = np.dot(A, s.base_vertices) + d
                base_vectors = np.dot(A, s.base_vectors)
                vals = base_vertices - np.sum(np.abs(base_vectors),axis=1)
                if np.all(vals>0):
                    safe[indx] = True
                    break

            if not safe[indx]: break

        return np.all(safe)


    def verify(self, s):
        unsafe = False
        for ud in self.unsafe_domains:
            A_unsafe = ud[0]
            d_unsafe = ud[1]
            if len(A_unsafe) == 1:
                vertices = np.dot(s.vertices, s.M.T) + s.b.T
                vals = np.dot(A_unsafe, vertices.T) + d_unsafe
                if np.any(np.all(vals<=0, axis=0)):
                    unsafe = True
                    break
            else:
                unsafe = self.backtrack(s, verify=True, unsafe_domain=ud)
                if unsafe:
                    break

        return unsafe


    def compute_state(self, tuple_state):
        s, layer, neurons = tuple_state # (vfl, layer, neurons)

        if (layer == self._num_layer - 1) and (len(neurons)==0):  # the last layer
            return [(s, layer, np.array([]))]

        new_tuple_states = []
        if neurons.shape[0] == 0: # neurons empty, go to the next layer
            if self.linearization or self.repair:
                over_app_set = self.reach_over_app(tuple_state)
                if self.verify_vzono(over_app_set):
                    return []

            W = self._W[layer+1]
            b = self._b[layer+1]
            s.affineMap(W, b)
            new_tuple_states.append((s, layer+1, np.arange(s.M.shape[0])))
        else: # not empty
            new_vfl_sets, new_neurons = relu.exact_reach(s, neurons)
            for vfl in new_vfl_sets:
                new_tuple_states.append((vfl, layer, new_neurons))

        return new_tuple_states






