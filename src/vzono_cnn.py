import torch
import numpy as np

class Vzono:
    def __init__(self, lbs, ubs):
        self.base_vertices = (lbs + ubs)/2
        self.base_vectors = self.create_base_vectors(lbs, ubs)


    def create_base_vectors(self, lbs, ubs):
        shape = list(lbs.size())
        shape[0] = np.prod(list(lbs.size()))
        base_vectors = torch.zeros(tuple(shape))
        vectors = (ubs - lbs)/2
        dim0 = 0
        for dim1 in range(shape[1]):
            for dim2 in range(shape[2]):
                for dim3 in range(shape[3]):
                    base_vectors[dim0, dim1, dim2, dim3] = vectors[0, dim1, dim2, dim3]
                    dim0 += 1

        # base_vectors = base_vectors.to_sparse()
        return base_vectors





