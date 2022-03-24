
import copy
import numpy as np
import collections as cln

class facelattice:
    def __init__(self, lattice, vertices, vertices_init, dim):
        self.lattice = lattice
        self.vertices = vertices
        self.vertices_init = vertices_init
        self.dim = dim
        self.fast_reach=False
        self.critical_v = []


    def to_cuda(self):
        self.is_cuda = True
        self.vertices = self.vertices.cuda()
        self.vertices_init = self.vertices_init.cuda()


    #conv
    def conv(self, nnet, layer):
        temp = nnet.features[layer](self.vertices)
        self.vertices = temp[:,:,1:-2,1:-2]


    def single_split_maxpool(self, idx):
        idx_pos = idx[0]
        idx_neg = idx[1]
        elements0 = self.vertices[:, idx_pos[0], idx_pos[1], idx_pos[2]]
        elements1 = self.vertices[:, idx_neg[0], idx_neg[1], idx_neg[2]]
        elements = elements0 - elements1
        if np.all(elements<=0):
            self.fast_reach=False
            return None, self
        elif np.all(elements>=0):
            self.fast_reach = False
            return self, None
        positive_bool = (elements>0)
        positive_id = np.asarray(positive_bool.nonzero()).T
        positive_vs = self.vertices[positive_bool]
        negative_bool = (elements<0)
        negative_id = np.asarray(negative_bool.nonzero()).T
        negative_vs = self.vertices[negative_bool]
        if len(positive_id)>=len(negative_id):
            less_vs_id = negative_id
            sign_flg = 1
        else:
            less_vs_id = positive_id
            sign_flg = -1
        edges_inter = []
        vs_new = np.zeros(0)
        vs_init_new = np.zeros(0)
        ref_vertex_list= list(self.lattice[0].keys())
        for p0_id in less_vs_id:
            edges = list(self.lattice[0].values())[p0_id[0]][1]
            p0 = self.vertices[p0_id]
            p0_init = self.vertices_init[p0_id]
            elem0 = p0[:,idx_pos[0],idx_pos[1],idx_pos[2]] - p0[:,idx_neg[0],idx_neg[1],idx_neg[2]]
            for ed in edges:
                for p1_rf in self.lattice[1][ed][0]:
                    p1_id = np.array([ref_vertex_list.index(p1_rf)])
                    p1 = self.vertices[p1_id]
                    sign_p1 = np.sign(p1[:,idx_pos[0],idx_pos[1],idx_pos[2]] - p1[:,idx_neg[0],idx_neg[1],idx_neg[2]])
                    if sign_p1 == sign_flg or sign_p1==0:
                        edges_inter.append(ed) # add the edge that spans negative and positive domain
                        p1_init = self.vertices_init[p1_id]
                        elem1 = p1[:,idx_neg[0],idx_neg[1],idx_neg[2]] - p1[:,idx_pos[0],idx_pos[1],idx_pos[2]]
                        alpha = abs(elem0)/(abs(elem0)+abs(elem1))
                        p_new = p0 + (p1-p0)*alpha
                        p_new_init = p0_init + (p1_init-p0_init)*alpha
                        if vs_new.shape[0]!=0:
                            vs_new = np.concatenate((vs_new, p_new), 0)
                            vs_init_new = np.concatenate((vs_init_new, p_new_init), 0)
                        else:
                            vs_new = p_new
                            vs_init_new = p_new_init

        # merge positive vertices with new_vs
        vs_new[:, idx[0][0], idx[0][1], idx[0][2]] = vs_new[:, idx[1][0], idx[1][1], idx[1][2]]
        negative_vs_init = self.vertices_init[negative_bool]
        positive_vs_init = self.vertices_init[positive_bool]
        # merge vertices with new_vs
        self.vertices = np.concatenate((self.vertices, vs_new), 0)
        self.vertices_init = np.concatenate((self.vertices_init, vs_init_new), 0)
        self.inter_lattice(edges_inter)
        if (sign_flg <= 0) or (not self.fast_reach):
            if self.fast_reach:
                positive_fl = []
            neg_vertices = np.concatenate((negative_vs, vs_new), 0)
            neg_vertices_init = np.concatenate((negative_vs_init, vs_init_new), 0)
            zero_id = np.arange(self.vertices.shape[0] - vs_new.shape[0], self.vertices.shape[0])
            negative_zero_id = np.concatenate((negative_id[:, 0], zero_id), 0)
            neg_lattice = self.split_lattice(negative_zero_id)
            negative_fl = facelattice(neg_lattice, neg_vertices, neg_vertices_init, self.dim)
        if (sign_flg > 0) or (not self.fast_reach):
            if self.fast_reach:
                negative_fl = []
            pos_vertices = np.concatenate((positive_vs, vs_new), 0)
            pos_vertices_init = np.concatenate((positive_vs_init, vs_init_new), 0)
            zero_id = np.arange(self.vertices.shape[0] - vs_new.shape[0], self.vertices.shape[0])
            positive_zero_id = np.concatenate((positive_id[:, 0], zero_id), 0)
            pos_lattice = self.split_lattice(positive_zero_id)
            positive_fl = facelattice(pos_lattice, pos_vertices, pos_vertices_init, self.dim)
        if sign_flg <= 0 and (not self.fast_reach):
            positive_fl.lattice = copy.deepcopy(positive_fl.lattice)
        if sign_flg > 0 and (not self.fast_reach):
            negative_fl.lattice = copy.deepcopy(negative_fl.lattice)
        return positive_fl, negative_fl


    def single_split_relu1(self, idx):
        elements = self.vertices[:,idx[0],idx[1],idx[2]]
        positive_bool = (elements>0)
        positive_id = np.asarray(positive_bool.nonzero()).T
        positive_vs = self.vertices[positive_bool]
        negative_bool = (elements<0)
        negative_id = np.asarray(negative_bool.nonzero()).T
        negative_vs = self.vertices[negative_bool]
        if positive_vs.shape[0]>=negative_vs.shape[0]:
            less_vs_id = negative_id
            sign_flg = 1
        else:
            less_vs_id = positive_id
            sign_flg = -1
        edges_inter = []
        vs_new = np.zeros(0)
        vs_init_new = np.zeros(0)
        ref_vertex_list= list(self.lattice[0].keys())
        for p0_id in less_vs_id:
            edges = list(self.lattice[0].values())[p0_id[0]][1]
            p0 = self.vertices[p0_id]
            p0_init = self.vertices_init[p0_id]
            elem0 = p0[:,idx[0],idx[1],idx[2]]
            for ed in edges:
                for p1_rf in self.lattice[1][ed][0]:
                    p1_id = np.array([ref_vertex_list.index(p1_rf)])
                    p1 = self.vertices[p1_id]
                    sign_p1 = np.sign(p1[:,idx[0],idx[1],idx[2]])
                    if  sign_p1 == sign_flg or sign_p1 == 0:
                        edges_inter.append(ed) # add the edge that spans negative and positive domain
                        p1_init = self.vertices_init[p1_id]
                        elem1 = p1[:,idx[0],idx[1],idx[2]]
                        alpha = abs(elem0)/(abs(elem0)+abs(elem1))
                        p_new = p0 + (p1-p0)*alpha
                        p_new_init = p0_init + (p1_init-p0_init)*alpha
                        if vs_new.shape[0]!=0:
                            vs_new = np.concatenate((vs_new, p_new), 0)
                            vs_init_new = np.concatenate((vs_init_new, p_new_init), 0)
                        else:
                            vs_new = p_new
                            vs_init_new = p_new_init

        # merge positive vertices with new_vs
        vs_new[:, idx[0], idx[1], idx[2]] = 0.0
        negative_vs_init = self.vertices_init[negative_bool]
        positive_vs_init = self.vertices_init[positive_bool]
        self.vertices = np.concatenate((self.vertices, vs_new), 0)
        self.vertices_init = np.concatenate((self.vertices_init, vs_init_new), 0)
        self.inter_lattice(edges_inter)

        zero_id = np.arange(self.vertices.shape[0] - vs_new.shape[0], self.vertices.shape[0])
        negative_zero_id = np.concatenate((negative_id[:, 0], copy.copy(zero_id)), 0)
        positive_zero_id = np.concatenate((positive_id[:, 0], copy.copy(zero_id)), 0)

        if (sign_flg <= 0) or (not self.fast_reach):
            if self.fast_reach:
                positive_fl = []
            neg_vertices = np.concatenate((negative_vs, vs_new), 0)
            neg_vertices_init = np.concatenate((negative_vs_init, vs_init_new), 0)
            neg_lattice = self.split_lattice(negative_zero_id)

            negative_fl = facelattice(neg_lattice, neg_vertices, neg_vertices_init, self.dim)
            negative_fl.map_negative_fl_relu1(idx)
        if (sign_flg > 0) or (not self.fast_reach):
            if self.fast_reach:
                negative_fl = []
            pos_vertices = np.concatenate((positive_vs, vs_new), 0)
            pos_vertices_init = np.concatenate((positive_vs_init, vs_init_new), 0)
            pos_lattice = self.split_lattice(positive_zero_id)
            positive_fl = facelattice(pos_lattice, pos_vertices, pos_vertices_init, self.dim)
        if sign_flg <= 0 and (not self.fast_reach):
            positive_fl.lattice = copy.deepcopy(positive_fl.lattice)

        if sign_flg > 0 and (not self.fast_reach):
            negative_fl.lattice = copy.deepcopy(negative_fl.lattice)

        return positive_fl, negative_fl


    def single_split_relu2(self, idx):
        elements = self.vertices[:,idx]
        positive_bool = (elements>0)
        positive_id = np.asarray(positive_bool.nonzero()).T
        positive_vs = self.vertices[positive_bool]
        negative_bool = (elements<0)
        negative_id = np.asarray(negative_bool.nonzero()).T
        negative_vs = self.vertices[negative_bool]
        if positive_vs.shape[0]>=negative_vs.shape[0]:
            less_vs_id = negative_id
            sign_flg = 1
        else:
            less_vs_id = positive_id
            sign_flg = -1

        edges_inter = []
        vs_new = np.zeros(0)
        vs_init_new = np.zeros(0)
        ref_vertex_list= list(self.lattice[0].keys())
        for p0_id in less_vs_id:
            edges = list(self.lattice[0].values())[p0_id[0]][1]
            p0 = self.vertices[p0_id]
            p0_init = self.vertices_init[p0_id]
            elem0 = p0[:,idx]
            for ed in edges:
                for p1_rf in self.lattice[1][ed][0]:
                    p1_id = np.array([ref_vertex_list.index(p1_rf)])
                    p1 = self.vertices[p1_id]
                    sign_p1 = np.sign(p1[:,idx])
                    if sign_p1 == sign_flg or sign_p1 == 0:
                        edges_inter.append(ed) # add the edge that spans negative and positive domain
                        p1_init = self.vertices_init[p1_id]
                        elem1 = p1[:,idx]
                        alpha = abs(elem0)/(abs(elem0)+abs(elem1))
                        p_new = p0 + (p1-p0)*alpha
                        p_new_init = p0_init + (p1_init-p0_init)*alpha
                        if vs_new.shape[0]!=0:
                            vs_new = np.concatenate((vs_new, p_new), 0)
                            vs_init_new = np.concatenate((vs_init_new, p_new_init), 0)
                        else:
                            vs_new = p_new
                            vs_init_new = p_new_init

        # merge positive vertices with new_vs
        vs_new[:, idx] = 0.0
        negative_vs_init = self.vertices_init[negative_bool]
        positive_vs_init = self.vertices_init[positive_bool]
        self.vertices = np.concatenate((self.vertices, vs_new), 0)
        self.vertices_init = np.concatenate((self.vertices_init, vs_init_new), 0)
        self.inter_lattice(edges_inter)

        if (sign_flg <= 0) or (not self.fast_reach):
            if self.fast_reach:
                positive_fl = []
            neg_vertices = np.concatenate((negative_vs, vs_new), 0)
            neg_vertices_init = np.concatenate((negative_vs_init, vs_init_new), 0)
            zero_id = np.arange(self.vertices.shape[0] - vs_new.shape[0], self.vertices.shape[0])
            negative_zero_id = np.concatenate((negative_id[:, 0], zero_id), 0)
            neg_lattice = self.split_lattice(negative_zero_id)
            negative_fl = facelattice(neg_lattice, neg_vertices, neg_vertices_init, self.dim)
            negative_fl.map_negative_fl_relu2(idx)

        if (sign_flg > 0) or (not self.fast_reach):
            if self.fast_reach:
                negative_fl = []
            pos_vertices = np.concatenate((positive_vs, vs_new), 0)
            pos_vertices_init = np.concatenate((positive_vs_init, vs_init_new), 0)
            zero_id = np.arange(self.vertices.shape[0] - vs_new.shape[0], self.vertices.shape[0])
            positive_zero_id = np.concatenate((positive_id[:, 0], zero_id), 0)
            pos_lattice = self.split_lattice(positive_zero_id)
            positive_fl = facelattice(pos_lattice, pos_vertices, pos_vertices_init, self.dim)

        if sign_flg <= 0 and (not self.fast_reach):
            positive_fl.lattice = copy.deepcopy(positive_fl.lattice)

        if sign_flg > 0 and (not self.fast_reach):
            negative_fl.lattice = copy.deepcopy(negative_fl.lattice)

        return positive_fl, negative_fl


    def map_negative_fl_relu1(self, idx):
        self.vertices[:, idx[0], idx[1], idx[2]] = 0.0


    def map_negative_fl_relu2(self, idx):
        self.vertices[:, idx] = 0.0


    def map_negative_fl_multi_relu1(self, idx):
        self.vertices[:, idx[:,0], idx[:,1], idx[:,2]] = 0.0


    def map_negative_fl_multi_relu2(self, idx):
        self.vertices[:, idx] = 0.0


    def inter_lattice(self, edges_inter):
        # get the lattice all the faces of which intersect with target hyperplane
        inter_faces = edges_inter
        inter_lattice_orig = self.extract_inter_lattice(inter_faces)
        # copy new_lattice with different addresses
        inter_lattice_new = copy.deepcopy(inter_lattice_orig)
        # create containment relation between
        for i in range(self.dim):
            rfs_list_orig = list(inter_lattice_orig[i].keys())
            rfs_list_new = list(inter_lattice_new[i].keys())
            for n in range(len(rfs_list_orig)):
                rf_orig = rfs_list_orig[n]
                rf_new = rfs_list_new[n]
                inter_lattice_new[i][rf_new][1].add(rf_orig)
                self.lattice[i + 1][rf_orig][0].add(rf_new)
        # merge self.lattice with inter_lattice
        for i in range(self.dim):
            self.lattice[i].update(inter_lattice_new[i])


    def split_lattice(self, selected_vs):
        # extract lattice according to vertices
        alattice = self.extract_vertex_lattice(selected_vs)
        return alattice


    def extract_vertex_lattice(self, idx):
        temp = list(self.lattice[0].keys())
        vertex = [temp[i] for i in idx]
        alattice = self.extract_lattice(vertex, 0)
        return alattice


    def extract_inter_lattice(self, rfs):
        alattice = self.extract_lattice(rfs, 1)
        # edit the bottom layer
        for key in alattice[0].keys():
            alattice[0][key][0] = set()
        # edit the top layer
        for key in alattice[self.dim-1].keys():
            alattice[self.dim-1][key][1] = set()
        return alattice


    def extract_lattice(self, rfs, n):
        alattice = []
        last_aset_temp = set()
        rfs_temp = rfs
        for i in range(n, self.dim+1):
            dict_temp = cln.OrderedDict()
            next_aset_temp = set()
            for rf in rfs_temp:
                rf_values =  copy.copy(self.lattice[i][rf])
                rf_values[0] =rf_values[0]&last_aset_temp
                dict_temp.update({rf: rf_values})
                next_aset_temp = next_aset_temp.union(rf_values[1])
            alattice.append(dict_temp)
            last_aset_temp = set(rfs_temp)
            rfs_temp = next_aset_temp
        # self.test_face_num(alattice)
        return alattice

    def test_face_num(self, hss):
        for m in range(self.dim-1):
            if m + 1 > len(hss):
                break
            face_m0 = hss[m]
            face_m1 = hss[m + 1]
            f1_f0 = set()
            for k in hss[m + 1].keys():
                f1_f0.update(hss[m + 1][k][0])
            f0_f1 = set()
            for k in hss[m].keys():
                f0_f1.update(hss[m][k][1])
            if len(f1_f0) != len(face_m0):
                print('f1_f0', m)
            if len(f0_f1) != len(face_m1):
                print('f0_f1', m)

