
import numpy as np

class Vzono:
    def __init__(self, base_vertices: np.ndarray, base_vectors: np.ndarray):
        self.base_vertices = base_vertices
        self.base_vectors = base_vectors

    def affineMap(self, W: np.ndarray, b: np.ndarray):
        self.base_vertices = np.dot(W, self.base_vertices) + b
        self.base_vectors = np.dot(W, self.base_vectors)

    def affineMapNegative(self, neurons_neg:np.ndarray):
        self.base_vertices[neurons_neg,:] = 0.0
        self.base_vectors[neurons_neg,:] = 0.0

    def reluLinearRelax(self, neurons_neg_pos:np.ndarray):
        assert neurons_neg_pos.shape[0] != 0

        # compute ubs and lbs of relu neurons
        base_vectices = self.base_vertices[neurons_neg_pos,:]
        vals = np.sum(np.abs(self.base_vectors[neurons_neg_pos,:]), axis=1, keepdims=True)
        ubs = np.max(base_vectices,axis=1, keepdims=True) + vals
        lbs = np.min(base_vectices, axis=1, keepdims=True) - vals

        A = ubs / (ubs - lbs)
        epsilons = -lbs*A/2
        M = np.eye(self.base_vertices.shape[0])
        d = np.zeros((self.base_vertices.shape[0], 1))
        M[neurons_neg_pos, neurons_neg_pos] = A[:,0]
        d[neurons_neg_pos] = epsilons

        base_vectors_relax = np.zeros((self.base_vertices.shape[0], len(neurons_neg_pos)))
        base_vectors_relax[neurons_neg_pos, range(len(ubs))] = epsilons[:,0]

        new_base_vertices = np.dot(M,self.base_vertices) + d
        new_base_vectors = np.concatenate((np.dot(M, self.base_vectors), base_vectors_relax), axis=1)
        self.base_vertices = new_base_vertices
        self.base_vectors = new_base_vectors


