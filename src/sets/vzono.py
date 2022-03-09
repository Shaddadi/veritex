
import torch
import numpy as np
import time

class VzonoAdap: # adaptive over approximation
    def __init__(self, base_vertices, matrix_A, vector_v):
        self.base_vertices = base_vertices
        self.matrix_A = [matrix_A]
        self.vector_v = [vector_v]

    def affine_mapping(self, W, b):
        self.base_vertices = np.dot(W, self.base_vertices) + b
        for n in range(len(self.matrix_A)):
            self.matrix_A[n] = np.dot(W, self.matrix_A[n])
            self.vector_v[n] = np.dot(W, self.vector_v[n])

    def affine_mapping_negative(self, neurons_neg):
        self.base_vertices[neurons_neg, :] = 0.0
        for n in range(len(self.matrix_A)):
            self.matrix_A[n][neurons_neg, :] = 0.0
            self.vector_v[n][neurons_neg, :] = 0.0


    def compute_interval(self):
        interval_vals = np.dot(self.matrix_A, self.base_vertices) + self.vector_v

    def relu_linear_relax_adaptive(self, neurons_neg_pos):
        assert neurons_neg_pos.shape[0] != 0
        base_vectices = self.base_vertices[neurons_neg_pos, :]
        # vals = np.





class VzonoFFNN:
    def __init__(self, base_vertices: np.ndarray, base_vectors: np.ndarray):
        self.base_vertices = base_vertices
        self.base_vectors = base_vectors

    def create_from_bounds(self, lbs, ubs):
        self.base_vertices = (np.array(lbs)+np.array(ubs))/2
        self.base_vectors = np.diag((np.array(ubs)-np.array(lbs))/2)

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



class Vzonop:
    def __init__(self, lbs, ubs, layer_data, split_dim):
        self.base_vertices = (lbs + ubs) / 2
        self.split_dim = split_dim[0]
        self.layer_data = layer_data
        dim = split_dim[0][1]
        self.base_vector_bias = torch.zeros(lbs.shape)
        self.base_vector_bias[0,dim[0],dim[1],dim[2]] = (ubs[:,dim[0],dim[1],dim[2]] - lbs[:,dim[0],dim[1],dim[2]])/2



class Vzono:
    def __init__(self, lbs, ubs, is_cuda=False, test=False, sparse=False):
        self.sparse = sparse
        if sparse:
            base_vertices = (lbs + ubs)/2
            self.base_vertices = spconv.SparseConvTensor.from_dense(base_vertices.permute(0, 2, 3, 1))
        else:
            self.base_vertices = (lbs + ubs)/2

        if test:
            self.base_vectors = self.create_base_vectors2d(lbs, ubs)
        else:
            self.base_vectors = self.create_base_vectors3d(lbs, ubs)


    def to_cuda(self):
        self.base_vertices = self.base_vertices.cuda()
        self.base_vectors = self.base_vectors.cuda()


    def to_sparse(self):
        t0 = time.time()
        self.base_vertices = [v.to_sparse() for v in self.base_vertices]
        self.base_vectors = [v.to_sparse() for v in self.base_vectors]

    def create_base_vectors3d(self, lbs, ubs):
        shape = list(lbs.size())
        shape[0] = np.prod(list(lbs.size()))
        base_vectors = torch.zeros(tuple(shape))

        dim0 = 0
        vectors = (ubs - lbs) / 2
        for dim1 in range(shape[1]):
            for dim2 in range(shape[2]):
                for dim3 in range(shape[3]):
                    base_vectors[dim0, dim1, dim2, dim3] = vectors[0, dim1, dim2, dim3]
                    dim0 += 1

        if self.sparse:
            return spconv.SparseConvTensor.from_dense(base_vectors.permute(0, 2, 3, 1))
        else:
            return base_vectors




    def create_base_vectors2d(self, lbs, ubs):
        shape = list(lbs.size())
        shape[0] = np.prod(list(lbs.size()))
        base_vectors = torch.zeros(tuple(shape))

        dim0 = 0
        vectors = (ubs - lbs) / 2
        for dim1 in range(shape[1]):
            base_vectors[dim0, dim1] = vectors[0, dim1]
            dim0 += 1
        # base_vectors = base_vectors.to_sparse()
        return base_vectors







