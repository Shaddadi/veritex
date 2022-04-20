import sys
import itertools
import numpy as np
import operator as op
from functools import reduce
import facelattice as fl
import collections as cln

class cubelattice:
    """
    A class for the set representation of a cube based on Face Lattice

        Attributes:
            dim (int): Dimensionality of the cube
            bs (np.ndarray): Intervals of dimensions
            lb (list): Lower bound of the cube
            ub (list): Upper bound of the cube
            vertices (np.ndarray): Vertices of the cube
            lattice (list): Containment relation bewteen (k)-dim faces and (k-1)-dim faces
            id_vals
            vertices_ref
            ref_vertex
    """
    def __init__(self, range_set):
        self.dim = len(range_set)
        self.bs = np.array(range_set)
        self.lb = self.bs[:,0].tolist()
        self.ub = self.bs[:,1].tolist()

        self.vertices = self.compute_vertex(self.lb, self.ub)
        self.lattice, self.id_vals, self.vertices_ref, self.ref_vertex = self.initial_lattice()
        for m in range(1,self.dim):
            self.single_dim_face(m)


    def to_facelattice(self, sp): #shape height x weight
        self.vertices = self.vertices.reshape((self.vertices.shape[0], 3, sp[1][0]-sp[0][0]+1, sp[1][1]-sp[0][1]+1))
        return fl.facelattice(self.lattice, self.vertices, self.vertices, self.dim)


    def initial_lattice(self):
        lattice = []
        id_vals = []
        vertex_ref = cln.OrderedDict()
        ref_vertex = cln.OrderedDict()
        n = self.dim
        for m in range(self.dim):
            num = 2**(n-m)*self.ncr(n,m)
            d = cln.OrderedDict()
            val = cln.OrderedDict()
            for i in range(num):
                id = reference(i)
                d.update({id:[set(),set()]})
                val.update({id: [[],[]]})
                if m == 0:
                    vertex_ref.update({tuple(self.vertices[i]):id})
                    ref_vertex.update({id: self.vertices[i]})
            lattice.append(d)
            id_vals.append([])

        # self.dim level
        id = reference(-1)
        for key in lattice[-1].keys():
            lattice[-1][key][1].add(id)
        lattice.append(cln.OrderedDict({id:[set(list(lattice[-1].keys())), set()]}))

        return lattice, id_vals, vertex_ref, ref_vertex


    def compute_vertex(self, lb, ub):
        # compute vertex
        V = []
        for i in range(len(ub)):
            V.append([lb[i], ub[i]])

        return np.array(list(itertools.product(*V)))


    # update lattice of m_face
    def single_dim_face(self, m):
        num = 2 ** (self.dim - m) * self.ncr(self.dim, m)
        Varray = self.vertices
        ref_m = list(self.lattice[m].keys())
        ref_m_1 = list(self.lattice[m-1].keys())

        id_vals_temp = cln.OrderedDict()

        nlist = list(range(len(self.lb)))
        element_id_sets = list(itertools.combinations(nlist, self.dim-m))
        c = 0
        for element_id in element_id_sets:
            # start_time = time.time()
            elem_id_m = np.array(element_id)
            vals = [list(self.bs[e,:]) for e in elem_id_m]
            faces = np.array(list(itertools.product(*vals)))

            diff_elem = np.setdiff1d(np.array(range(self.dim)), elem_id_m)

            for f in faces:
                f_m = np.ones((self.dim))*100
                for i in range(len(elem_id_m)):
                    f_m[elem_id_m[i]] = f[i]
                k_m = tuple(np.concatenate((elem_id_m, f_m)))
                id_m = ref_m[c]
                id_vals_temp.update({k_m: id_m})

                for i in diff_elem:
                    elem_id_m_1 = np.copy(elem_id_m)
                    elem_id_m_1 = np.sort(np.append(elem_id_m_1, i))
                    f_m_1 = np.copy(f_m)
                    # upper bound
                    f_m_1[i] = self.ub[i]
                    k_m_1 = tuple(np.concatenate((elem_id_m_1, f_m_1)))
                    if m!=1:
                        id_m_1 = self.id_vals[m - 1][k_m_1]
                    else:
                        id_m_1 = self.vertices_ref[tuple(f_m_1)]

                    self.lattice[m][ref_m[c]][0].add(id_m_1)
                    self.lattice[m - 1][id_m_1][1].add(ref_m[c])

                    # lower bound
                    f_m_1[i] = self.lb[i]
                    k_m_1 = tuple(np.concatenate((elem_id_m_1, f_m_1)))
                    if m != 1:
                        id_m_1 = self.id_vals[m - 1][k_m_1]
                    else:
                        id_m_1 = self.vertices_ref[tuple(f_m_1)]

                    self.lattice[m][ref_m[c]][0].add(id_m_1)
                    self.lattice[m - 1][id_m_1][1].add(ref_m[c])

                c = c+1

        self.id_vals[m] = id_vals_temp

        if c!=num:
            print('Computation is wrong')
            sys.exit(1)


    def ncr(self, n, r):
        r = min(r, n-r)
        numer = reduce(op.mul, range(n, n-r, -1), 1)
        denom = reduce(op.mul, range(1, r+1), 1)
        return int(numer / denom)


class reference:
    def __init__(self, val):
        self._value = val  # just refers to val, no copy


    # print(hull.vertices)