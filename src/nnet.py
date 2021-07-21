import numpy as np
import time
import os
import sys
import psutil
import pickle
import multiprocessing
from functools import partial
from multiprocessing import get_context
import torch
import copy as cp

class nnetwork:

    def __init__(self, W, b,properties=None):
        self.W = W
        self.b = b
        self.c = 0
        self.numLayer = len(W)
        self.properties = properties


    def backtrack(self, vfl_set):
        # print('backtrack, here')
        matrix_A = self.properties[1][0]
        vector_d = self.properties[1][1]

        elements = np.dot(np.dot(matrix_A,vfl_set.M), vfl_set.vertices.T) + np.dot(matrix_A, vfl_set.b) +vector_d
        unsafe_vs = np.all(elements<0, axis=0)
        unsafe_vs_num = len(np.nonzero(unsafe_vs)[0])
        unsafe_ratio = unsafe_vs_num / len(vfl_set.vertices)
        if unsafe_vs_num == 0:
            return None, None, 0

        vfls_unsafe = cp.deepcopy(vfl_set)
        for n in range(len(matrix_A)):
            A = matrix_A[[n]]
            d = vector_d[[n]]
            subvfl0 = vfls_unsafe.single_split(A, d)
            if subvfl0:
                vfls_unsafe = subvfl0
            else:
                vfls_unsafe = None
                break

        if vfls_unsafe:
            unsafe_data_final = vfls_unsafe.vertices[0]
        else:
            unsafe_data_final = None

        return unsafe_data_final, vfls_unsafe, unsafe_ratio


    # nn output of input starting from mth layer
    def layerOutput(self, inputPoly, layer=0, over_app=False):
        # print('Layer: %d\n'%m)
        output_data = []
        if layer==self.numLayer:
            data_unsafe, vfls_unsafe, ratio_unsafe = self.backtrack(inputPoly)
            return [[data_unsafe, vfls_unsafe, inputPoly]]

        if over_app and layer<self.numLayer-1:
            over_app_poly = cp.deepcopy(inputPoly)
            over_app_poly = self.layerOutputOverApp(over_app_poly, layer)
            safe = self.check_safety_over_app(over_app_poly)
            if safe:
                return []

        inputSets = self.singleLayerOutput(inputPoly, layer)
        for apoly in inputSets:
            output_data.extend(self.layerOutput(apoly, layer=layer+1, over_app=over_app))

        return output_data

    def layerOutputOverApp(self, apoly, layer):
        apoly.compute_real_vertices()

        for layerID in range(layer, self.numLayer):
            apoly = self.singleLayerOverApproximation(apoly, layerID)

        return apoly


    def check_safety_over_app(self, apoly):
        matrix_A = self.properties[1][0]
        vector_d = self.properties[1][1]
        safe = True
        for n in range(len(matrix_A)):
            A = matrix_A[[n]]
            d = vector_d[[n]]
            base_vertices = np.dot(A, apoly.base_vertices) + d
            base_vectors = np.dot(A, apoly.base_vectors)
            vals = base_vertices - np.sum(np.abs(base_vectors))
            if np.any(vals <= 0):
                return False

        return safe


    def outputPoint(self, inputPoint):
        for i in range(self.numLayer):
            inputPoint = self.singleLayerPointOutput(inputPoint, i)

        return inputPoint


    def singleLayerPointOutput(self, inputPoint, layerID):
        W = self.W[layerID]
        b = self.b[layerID]
        layerPoint = np.dot(W, inputPoint.transpose())+b
        if layerID == self.numLayer-1:
            return layerPoint.transpose()
        else:
            layerPoint[layerPoint<0] = 0
            return layerPoint.transpose()


    def outputPointBeforeReLU(self, inputPoint):
        for i in range(self.numLayer-1):
            PointBeforeReLU = self.singleLayerPointOutputBeforeRelU(inputPoint, i)

            inputPoint = cp.copy(PointBeforeReLU)
            inputPoint[PointBeforeReLU < 0] = 0

        outPoint = self.singleLayerPointOutputBeforeRelU(inputPoint, self.numLayer-1)

        return outPoint


    def singleLayerPointOutputBeforeRelU(self, inputPoint, layerID):
        W = self.W[layerID]
        b = self.b[layerID]
        PointBeforeReLU = np.dot(W, inputPoint.transpose())+b

        PointBeforeReLU_bool = PointBeforeReLU>=0.00001
        xx = np.nonzero(np.any(PointBeforeReLU_bool, axis=1))[0]

        return PointBeforeReLU.transpose()


    def singleLayerOutput(self, inputPoly, layerID):
        # print("layer", layerID)
        W = self.W[layerID]
        b = self.b[layerID]
        inputPoly.linearTrans(W, b)

        # partition graph sets according to properties of the relu function
        if layerID == self.numLayer-1:
            return [inputPoly]

        splited_polys = []
        splited_polys.extend(self.relu_layer(inputPoly, np.array([]), flag=False))

        return splited_polys


    def singleLayerOverApproximation(self, apoly, layerID):
        W = self.W[layerID]
        b = self.b[layerID]
        apoly.base_vertices = np.dot(W, apoly.base_vertices) + b
        apoly.base_vectors = np.dot(W, apoly.base_vectors)

        if layerID == self.numLayer-1:
            return apoly

        over_app_poly = self.relu_layer_over_app(apoly)

        return over_app_poly


    # partition one input polytope with a hyberplane
    def splitPoly(self, inputPoly, idx):
        outputPolySets = []
        sub0, sub1= inputPoly.single_split_relu(idx)
        if sub0:
            outputPolySets.append(sub0)
        if sub1:
            outputPolySets.append(sub1)

        return outputPolySets


    def relu_layer(self, im_fl, neurons, flag=True):
        if (neurons.shape[0] == 0) and flag:
            return [im_fl]

        new_neurons, new_neurons_neg = self.get_valid_neurons(im_fl, neurons)

        im_fl.map_negative_poly(new_neurons_neg)

        if new_neurons.shape[0] == 0:
            return [im_fl]

        fls = self.splitPoly(im_fl, new_neurons[0])
        new_neurons = new_neurons[1:]

        all_fls = []
        for afl in fls:
            all_fls.extend(self.relu_layer(afl, new_neurons))

        return all_fls


    def relu_layer_over_app(self, im_fl):
        neurons_neg_pos, neurons_neg = self.get_valid_neurons_for_over_app(im_fl)
        im_fl.base_vertices[neurons_neg,:] = 0
        im_fl.base_vectors[neurons_neg,:] = 0

        if neurons_neg_pos.shape[0] == 0:
            return im_fl

        base_vectices = im_fl.base_vertices[neurons_neg_pos,:]
        vals = np.sum(np.abs(im_fl.base_vectors[neurons_neg_pos,:]), axis=1, keepdims=True)
        ubs = np.max(base_vectices,axis=1, keepdims=True) + vals
        lbs = np.min(base_vectices, axis=1, keepdims=True) - vals
        M = np.eye(im_fl.base_vertices.shape[0])
        b = np.zeros((im_fl.base_vertices.shape[0], 1))
        base_vectors_relax = np.zeros((im_fl.base_vertices.shape[0],len(neurons_neg_pos)))

        A = ubs / (ubs - lbs)
        epsilons = -lbs*A/2
        M[neurons_neg_pos, neurons_neg_pos] = A[:,0]
        b[neurons_neg_pos] = epsilons
        base_vectors_relax[neurons_neg_pos, range(len(ubs))] = epsilons[:,0]

        new_base_vertices = np.dot(M,im_fl.base_vertices) + b
        new_base_vectors = np.concatenate((np.dot(M, im_fl.base_vectors), base_vectors_relax), axis=1)
        im_fl.base_vertices = new_base_vertices
        im_fl.base_vectors = new_base_vectors

        return im_fl


    def get_valid_neurons(self, afl, neurons):
        if neurons.shape[0] ==0:
            vertices = np.dot(afl.vertices, afl.M.T) + afl.b.T
            flag_neg = vertices<=0
            temp_neg = np.all(flag_neg, 0)
            valid_neurons_neg = np.asarray(np.nonzero(temp_neg)).T[:,0]
            temp_pos = np.all(vertices>=0, 0)
            neurons_sum = temp_neg+temp_pos
            valid_neurons_neg_pos = np.asarray(np.nonzero(neurons_sum==False)).T[:,0]
            return valid_neurons_neg_pos, valid_neurons_neg

        elements = np.dot(afl.vertices,afl.M[neurons,:].T)+afl.b[neurons,:].T
        flag_neg = (elements <= 0)
        temp_neg = np.all(flag_neg, 0)
        temp_pos = np.all(elements>=0, 0)
        temp_sum = temp_neg + temp_pos
        indx_neg_pos = np.asarray(np.nonzero(temp_sum == False)).T[:,0]
        valid_neurons_neg_pos = neurons[indx_neg_pos]
        indx_neg = np.asarray(np.nonzero(temp_neg)).T[:,0]
        valid_neurons_neg = neurons[indx_neg]

        return valid_neurons_neg_pos, valid_neurons_neg


    def get_valid_neurons_for_over_app(self, afl):
        vals = np.sum(np.abs(afl.base_vectors), axis=1, keepdims=True)
        temp_neg = np.all((afl.base_vertices+vals) <= 0, 1)
        valid_neurons_neg = np.asarray(np.nonzero(temp_neg)).T[:, 0]
        temp_pos = np.all((afl.base_vertices-vals) >= 0, 1)
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
