import torch
import numpy as np
import time
import spconv



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





