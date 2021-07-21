import sys
import itertools
import numpy as np
import vertex_facet_lattice as vfl


class cubelattice:

    def __init__(self, lb, ub):
        self.dim = 9
        ls = [0,1,2,3,4,5,8,9,10]
        self.lb = [lb[i] for i in ls]
        self.ub = [ub[i] for i in ls]
        self.M = np.eye(len(lb))
        self.b = np.zeros((len(lb),1))
        self.bs = np.array([self.lb,self.ub]).T

        self.vertices = self.compute_vertex(self.lb, self.ub)
        self.compute_lattice() # compute self.lattice

    def to_lattice(self):
        self.vertices = np.concatenate([self.vertices[:, :6], np.zeros((len(self.vertices), 2)), self.vertices[:, 6:]], axis=1)
        return vfl.VFL(self.lattice, self.vertices, self.dim, self.M, self.b)

    def compute_lattice(self):
        vertex_facets = []
        for idx, vals in enumerate(self.bs):
            for val in vals:
                vs_facet = self.vertices[:,idx]==val
                vertex_facets.append(vs_facet)

        self.lattice = np.array(vertex_facets).transpose()


    def compute_vertex(self, lb, ub):
        V = []
        for i in range(len(ub)):
            V.append([lb[i], ub[i]])


        vertices = np.array(list(itertools.product(*V)))
        return vertices

