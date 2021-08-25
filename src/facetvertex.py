import sys
import copy as cp
import numpy as np
import collections as cln

class FacetVertex:
    def __init__(self, fmatrix:np.ndarray, vertices:np.ndarray, dim: int, M:np.ndarray, b:np.ndarray):
        self.fmatrix = fmatrix
        self.vertices = vertices
        self.dim = dim
        self.M = M
        self.b = b


    def affineMap(self, M:np.ndarray, b:np.ndarray):
        self.M = np.dot(M, self.M)
        self.b = np.dot(M, self.b) + b



    def affineMapNegative(self, n:int):
        self.M[n, :] = 0
        self.b[n, :] = 0


    def reluSplit(self, neuron_pos_neg:np.ndarray):
        elements = np.matmul(self.vertices, self.M[neuron_pos_neg,:].T)+self.b[neuron_pos_neg,:].T
        if np.any(elements==0.0):
            sys.exit('Hyperplane intersect with vertices!')

        positive_bool = (elements>0)
        positive_id = np.asarray(positive_bool.nonzero()).T
        negative_bool = np.invert(positive_bool)
        negative_id = np.asarray(negative_bool.nonzero()).T

        if len(positive_id)>=len(negative_id):
            less_bool = negative_bool
            more_bool = positive_bool
            flg = 1
        else:
            less_bool = positive_bool
            more_bool = negative_bool
            flg = -1

        vs_facets0 = self.fmatrix[less_bool]
        vs_facets1 = self.fmatrix[more_bool]
        vertices0 = self.vertices[less_bool]
        vertices1 = self.vertices[more_bool]
        elements0 = elements[less_bool]
        elements1 = elements[more_bool]

        edges = np.dot(vs_facets0.astype(np.float32), vs_facets1.T.astype(np.float32))
        edges_indx = np.array(np.nonzero(edges == self.dim - 1))
        if len(edges_indx[0])+len(edges_indx[1]) == 0:
            sys.exit('Intersected edges are empty!')
        indx0, indx1 = edges_indx[0], edges_indx[1]
        p0s, p1s = vertices0[indx0], vertices1[indx1]
        elem0, elem1s = elements0[indx0], elements1[indx1]
        alpha = abs(elem0) / (abs(elem0) + abs(elem1s))

        new_vs = p0s + ((p1s - p0s).T * alpha).T
        new_vs_facets = np.logical_and(vs_facets0[indx0], vs_facets1[indx1])

        new_vs_facets0 = np.concatenate((vs_facets0, new_vs_facets))
        sub_vs_facets0 = new_vs_facets0[:,np.any(vs_facets0,0)]
        vs_facets_hp = np.zeros((len(sub_vs_facets0), 1), dtype=bool)
        vs_facets_hp[-len(new_vs):,0] = True # add hyperplane
        sub_vs_facets0 = np.concatenate((sub_vs_facets0, vs_facets_hp), axis=1)
        new_vertices0 = np.concatenate((vertices0, new_vs))
        subset0 = FacetVertex(sub_vs_facets0, new_vertices0, self.dim, cp.copy(self.M), cp.copy(self.b))
        if flg == 1:
            subset0.affineMapNegative(neuron_pos_neg)

        new_vs_facets1 = np.concatenate((vs_facets1, new_vs_facets))
        sub_vs_facets1 = new_vs_facets1[:, np.any(vs_facets1, 0)]
        vs_facets_hp = np.zeros((len(sub_vs_facets1), 1), dtype=bool)
        vs_facets_hp[-len(new_vs):,0] = True # add hyperplane
        sub_vs_facets1 = np.concatenate((sub_vs_facets1, vs_facets_hp), axis=1)
        new_vertices1 = np.concatenate((vertices1, new_vs))
        subset1 = FacetVertex(sub_vs_facets1, new_vertices1, self.dim, cp.copy(self.M), cp.copy(self.b))
        if flg == -1:
            subset1.affineMapNegative(neuron_pos_neg)
        
        return subset0, subset1


    def reluSplitHyperplane(self, A:np.ndarray, d:np.ndarray):
        A_new = np.dot(A,self.M)
        d_new = np.dot(A, self.b) +d
        elements = np.dot(A_new, self.vertices.T) + d_new
        elements = elements[0]
        if np.all(elements >= 0):
            return None
        if np.all(elements <= 0):
            return self
        if np.any(elements == 0.0):
            sys.exit('Hyperplane intersect with vertices!')

        positive_bool = (elements > 0)
        negative_bool = np.invert(positive_bool)

        vs_facets0 = self.fmatrix[negative_bool]
        vs_facets1 = self.fmatrix[positive_bool]
        vertices0 = self.vertices[negative_bool]
        vertices1 = self.vertices[positive_bool]
        elements0 = elements[negative_bool]
        elements1 = elements[positive_bool]

        # t0 = time.time()
        edges = np.dot(vs_facets0.astype(np.float32), vs_facets1.T.astype(np.float32))
        edges_indx = np.array(np.nonzero(edges == self.dim - 1))
        if len(edges_indx[0])+len(edges_indx[1]) == 0:
            sys.exit('Intersected edges are empty!')
        indx0, indx1 = edges_indx[0], edges_indx[1]
        p0s, p1s = vertices0[indx0], vertices1[indx1]

        elem0, elem1s = elements0[indx0], elements1[indx1]
        alpha = abs(elem0) / (abs(elem0) + abs(elem1s))
        new_vs = p0s + ((p1s - p0s).T * alpha).T
        new_vs_facets = np.logical_and(vs_facets0[indx0], vs_facets1[indx1])

        new_vs_facets0 = np.concatenate((vs_facets0, new_vs_facets))
        sub_vs_facets0 = new_vs_facets0[:, np.any(vs_facets0, 0)]
        vs_facets_hp = np.zeros((len(sub_vs_facets0), 1), dtype=bool)
        vs_facets_hp[-len(new_vs):, 0] = True  # add hyperplane
        sub_vs_facets0 = np.concatenate((sub_vs_facets0, vs_facets_hp), axis=1)
        new_vertices0 = np.concatenate((vertices0, new_vs))
        subset0 = FacetVertex(sub_vs_facets0, new_vertices0, self.dim, cp.copy(self.M), cp.copy(self.b))

        return subset0





