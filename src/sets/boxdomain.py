import sys
import itertools
import numpy as np
from facetvertex import FacetVertex


class BoxDomain:

    def __init__(self, lbs=list, ubs=list):
        assert len(lbs) == len(ubs)

        lbs, ubs = np.array(lbs), np.array(ubs)
        self.dims_active = np.nonzero(lbs != ubs)[0]
        self.dims_static = np.nonzero(lbs == ubs)[0]
        self.dim = len(self.dims_active)

        self.lbs_active = lbs[self.dims_active]
        self.ubs_active = ubs[self.dims_active]

        self.lbs_static = lbs[self.dims_static]
        self.ubs_static = ubs[self.dims_static]

        self.M = np.eye(len(lbs))
        self.b = np.zeros((len(lbs),1))
        self.vertices = self.computeVertices()


    def toFacetVertex(self):
        fmatrix = self.computeFmatrix()
        return FacetVertex(fmatrix, self.vertices, self.dim, self.M, self.b)

    def computeFmatrix(self):
        facets_vertex = []
        combs = np.array([self.lbs_active, self.ubs_active]).T
        for n, vals in enumerate(combs):
            indx = self.dims_active[n]
            for val in vals:
                vs_facet = self.vertices[:,indx]==val
                facets_vertex.append(vs_facet)

        fmatrix = np.array(facets_vertex).transpose() # fmatrix: vertices, facets
        return fmatrix


    def computeVertices(self):
        dim_vertices = len(self.dims_active) + len(self.dims_static)
        vertices = np.zeros((2**self.dim, dim_vertices))
        V = []
        for i in range(self.dim):
            V.append([self.lbs_active[i], self.ubs_active[i]])

        vertices[:, self.dims_active] = np.array(list(itertools.product(*V)))
        vertices[:, self.dims_static] = np.repeat([self.lbs_static], 2**self.dim, axis=0)

        return vertices

