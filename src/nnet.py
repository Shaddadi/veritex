import sys

import numpy as np
import copy as cp
from vzono import Vzono
from itertools import product
from scipy.optimize import linprog
from collections import deque
import multiprocessing
import os




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
                    os.system('pkill -9 python')
                    break
            else:
                unsafe_vfl = self.backtrack(vfl_set, verify=True)
                if unsafe_vfl is not None:
                    unsafe = True
                    os.system('pkill -9 python')
                    break

        return unsafe


    def collectResults(self, vfl_set):
        results = []
        if self.config_only_verify:
            results.append(self.verify(vfl_set))
        else:
            results.append([])

        if self.config_exact_output:
            results.append([])
        else:
            results.append([])

        if self.config_unsafe_input:
            results.append([self.backtrack(vfl_set), vfl_set])
        else:
            results.append([])

        return results


    def reach(self, vfl_set, start_layer=0):
        output_data = []
        if start_layer==self._num_layer:
            return [self.collectResults(vfl_set)]

        if self.config_relu_linear and start_layer<self._num_layer-1:
            over_app_set = self.reachOverApp(vfl_set, start_layer)
            safe = self.verifyVzono(over_app_set)
            if safe: return []

        input_sets = self.singleLayerOutput(vfl_set, start_layer)
        for aset in input_sets:
            output_data.extend(self.reach(aset, start_layer=start_layer+1))

        return output_data


    def reachOverApp(self, vfl_set, layer_id):
        base_vertices = np.dot(vfl_set.M, vfl_set.vertices.T) + vfl_set.b
        base_vectors = np.zeros((base_vertices.shape[0], 1))
        vzono_set = Vzono(base_vertices, base_vectors)

        for n in range(layer_id, self._num_layer):
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

        # safe = True
        # for ud in self.unsafe_domains:
        #     A_unsafe = ud[0]
        #     d_unsafe = ud[1]
        #     if len(A_unsafe)==1:
        #         base_vertices = np.dot(A_unsafe, vzono_set.base_vertices) + d_unsafe
        #         base_vectors = np.dot(A_unsafe, vzono_set.base_vectors)
        #         vals = base_vertices - np.sum(np.abs(base_vectors),axis=1)
        #         if np.any(vals<=0):
        #             safe = False
        #             return safe
        #     else:
        #         ubs = np.max(vzono_set.base_vertices,axis=1) + np.sum(np.abs(vzono_set.base_vectors),axis=1)
        #         lbs = np.min(vzono_set.base_vertices,axis=1) - np.sum(np.abs(vzono_set.base_vectors),axis=1)
        #         dim = vzono_set.base_vertices.shape[0]
        #         bounds = tuple([(lbs[n], ubs[n]) for n in range(len(ubs))])
        #         try:
        #             rel = linprog(np.zeros(dim), A_ub=A_unsafe, b_ub=d_unsafe,bounds=bounds,method='interior-point')
        #             if rel['success']:
        #                 safe = False
        #                 return safe
        #         except:
        #             safe = False
        #             return safe
        #
        # return safe
        # matrix_A = self.properties[1][0]
        # vector_d = self.properties[1][1]
        # safe = True
        # for n in range(len(matrix_A)):
        #     A = matrix_A[[n]]
        #     d = vector_d[[n]]
        #     base_vertices = np.dot(A, vzono_set.base_vertices) + d
        #     base_vectors = np.dot(A, vzono_set.base_vectors)
        #     vals = base_vertices - np.sum(np.abs(base_vectors))
        #     if np.any(vals <= 0):
        #         return False
        #
        # return safe


    def outputPoint(self, inputPoint):
        for i in range(self._num_layer):
            inputPoint = self.singleLayerPointOutput(inputPoint, i)

        return inputPoint


    def singleLayerPointOutput(self, inputPoint, layer_id):
        W = self._W[layer_id]
        b = self._b[layer_id]
        layerPoint = np.dot(W, inputPoint.transpose())+b
        if layer_id == self._num_layer-1:
            return layerPoint.transpose()
        else:
            layerPoint[layerPoint<0] = 0
            return layerPoint.transpose()


    def outputPointBeforeReLU(self, inputPoint):
        for i in range(self._num_layer-1):
            PointBeforeReLU = self.singleLayerPointOutputBeforeRelU(inputPoint, i)

            inputPoint = cp.copy(PointBeforeReLU)
            inputPoint[PointBeforeReLU < 0] = 0

        outPoint = self.singleLayerPointOutputBeforeRelU(inputPoint, self._num_layer-1)

        return outPoint


    def singleLayerPointOutputBeforeRelU(self, inputPoint, layer_id):
        W = self._W[layer_id]
        b = self._b[layer_id]
        PointBeforeReLU = np.dot(W, inputPoint.transpose())+b

        PointBeforeReLU_bool = PointBeforeReLU>=0.00001
        xx = np.nonzero(np.any(PointBeforeReLU_bool, axis=1))[0]

        return PointBeforeReLU.transpose()


    def singleLayerOutput(self, vfl_set, layer_id):
        # print("layer", layer_id)
        W = self._W[layer_id]
        b = self._b[layer_id]
        vfl_set.affineMap(W, b)

        # q = deque()
        # q.append(vfl_set)
        #
        # queue = multiprocessing.Queue()
        # queue.put(vfl_set)
        # queue.put(vfl_set)
        # xx = queue.get()
        # partition graph sets according to properties of the relu function
        if layer_id == self._num_layer-1:
            return [vfl_set]

        splited_sets = []
        splited_sets.extend(self.reluLayer(vfl_set, np.array([]), flag=False))

        return splited_sets


    def singleLayerOverApp(self, vzono_set, layer_id):
        W = self._W[layer_id]
        b = self._b[layer_id]
        vzono_set.base_vertices = np.dot(W, vzono_set.base_vertices) + b
        vzono_set.base_vectors = np.dot(W, vzono_set.base_vectors)

        if layer_id == self._num_layer-1:
            return vzono_set

        over_app_set = self.reluLayerLinearRelax(vzono_set)

        return over_app_set


    # partition one input polytope with a hyberplane
    def reluSplit(self, vfl_set, idx):
        outputPolySets = []
        sub0, sub1= vfl_set.reluSplit(idx)
        if sub0:
            outputPolySets.append(sub0)
        if sub1:
            outputPolySets.append(sub1)

        return outputPolySets


    def reluLayer(self, vfl_set, neurons, flag=True):
        if (neurons.shape[0] == 0) and flag:
            return [vfl_set]

        new_neurons, new_neurons_neg = self.get_valid_neurons(vfl_set, neurons)

        vfl_set.affineMapNegative(new_neurons_neg)

        if new_neurons.shape[0] == 0:
            return [vfl_set]

        vfl_sets = self.reluSplit(vfl_set, new_neurons[0])
        if len(new_neurons) == 1:
            xx = 1
        new_neurons = new_neurons[1:]

        all_sets = []
        for aset in vfl_sets:
            all_sets.extend(self.reluLayer(aset, new_neurons))

        return all_sets


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


    def get_valid_neurons(self, vfl_set, neurons):
        if neurons.shape[0] ==0:
            vertices = np.dot(vfl_set.vertices, vfl_set.M.T) + vfl_set.b.T
            flag_neg = vertices<=0
            temp_neg = np.all(flag_neg, 0)
            valid_neurons_neg = np.asarray(np.nonzero(temp_neg)).T[:,0]
            temp_pos = np.all(vertices>=0, 0)
            neurons_sum = temp_neg+temp_pos
            valid_neurons_neg_pos = np.asarray(np.nonzero(neurons_sum==False)).T[:,0]
            return valid_neurons_neg_pos, valid_neurons_neg

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


    def get_valid_neurons_for_over_app(self, vfl_set):
        vals = np.sum(np.abs(vfl_set.base_vectors), axis=1, keepdims=True)
        temp_neg = np.all((vfl_set.base_vertices+vals) <= 0, 1)
        valid_neurons_neg = np.asarray(np.nonzero(temp_neg)).T[:, 0]
        temp_pos = np.all((vfl_set.base_vertices-vals) >= 0, 1)
        neurons_sum = temp_neg + temp_pos
        valid_neurons_neg_pos = np.asarray(np.nonzero(neurons_sum == False)).T[:, 0]

        return valid_neurons_neg_pos, valid_neurons_neg

    # def __getstate__(self):
    #     self_dict = self.__dict__.copy()
    #     del self_dict['pool']
    #     return self_dict
    #
    # def __setstate__(self, state):
    #     self.__dict__.update(state)
