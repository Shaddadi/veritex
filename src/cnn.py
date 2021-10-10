import numpy as np
import copy as cp
from vzono_cnn import Vzono, Vzonop
import sys
import torch
import time
import spconv


class CNN:

    def __init__(self, sequential, is_cuda=False, sparse=False):
        if is_cuda:
            self.sequential = sequential.eval().cuda()
        else:
            self.sequential = sequential.eval()

        self.is_cuda = is_cuda
        self.sparse = sparse
        self.layer_num = len(sequential)
        self.relus_over = [None] * (self.layer_num)
        self.layer_old_inputs = [None] * (self.layer_num)
        self.layer_neurons = [None] * (self.layer_num) # over-approximated relus for future splitting
        self.layer_vectors_sum = [None] * (self.layer_num)
        self.collect_layer_flag = True
        self.layer_input_size = []
        self._layer = None
        self.search_depth = np.infty



    def verify_depth_first(self, inputs, layer_data=None, split_dims=None):
        if layer_data:
            result, layer_data = self.reach_over_appr_post(inputs, layer_data, split_dims)
        else:
            result, layer_data = self.reach_over_appr(inputs)

        if result:
            return

        subsets, split_dims = self.split_input([inputs], num=1)
        for sub in subsets:
            layer_data_new = cp.deepcopy(layer_data)
            self.verify_depth_first(sub, layer_data=layer_data_new, split_dims=split_dims)



    def reach_over_appr_post(self, inputs, layer_data, split_dim):
        image_lbs = inputs[0]
        image_ubs = inputs[1]
        vzonop_set = Vzonop(image_lbs, image_ubs, layer_data, split_dim)

        for self._layer in range(self.layer_num):
            type_name = type(self.sequential[self._layer]).__name__
            # ['Conv2d', 'Linear', 'ReLU', 'Flatten']
            if type_name == 'Conv2d':
                t0 = time.time()
                vzonop_set = self.conv2d_post(vzonop_set)
                print('Conv2d time: ', time.time() - t0)
            elif type_name == 'ReLU':
                t0 = time.time()
                if len(vzonop_set.base_vertices.shape) == 4:
                    vzonop_set = self.relu_over_appr_post(vzonop_set)
                elif len(vzonop_set.base_vertices.shape) == 2: # fully connected layers
                    vzonop_set = self.relu_over_appr2_post(vzonop_set)
                print('ReLU time: ', time.time() - t0)
            elif type_name == 'Linear':
                t0 = time.time()
                vzonop_set = self.linear_post(vzonop_set)
                print('Linear time: ', time.time() - t0)
            elif type_name == 'Flatten':
                vzonop_set = self.flatten_post(vzonop_set)
            else:
                sys.exit('This layer type is not supported yet!')




    def reach_over_appr_parallel(self, inputs):
        inputs_list = [inputs]
        last_out_lbub = []
        for _ in range(1000):
            collt = []
            last_lbub_temp = [] # for test
            for index, item in enumerate(inputs_list):
                result, lbub = self.reach_over_appr(item)
                self.collect_layer_flag = False
                if last_out_lbub:
                    last_lbub = last_out_lbub[index//2]
                    last_lb = last_lbub[0]
                    last_ub = last_lbub[1]
                    curr_lb = lbub[0]
                    curr_ub = lbub[1]
                    diff_lb = curr_lb - last_lb
                    diff_ub = last_ub - curr_ub
                    assert torch.all(diff_lb>=0)
                    assert torch.all(diff_ub>=0)
                    assert torch.sum(diff_lb) > 0
                    assert torch.sum(diff_ub) > 0
                if not result: #unknown
                    collt.append(item)
                    last_lbub_temp.append(lbub)
            if collt:
                inputs_list = self.split_input(collt, num=1)
                last_out_lbub = cp.deepcopy(last_lbub_temp)
            else:
                break

        print('Here')
        if not inputs_list:
            return True # safe
        else:
            return False # unknown



    def reach_over_appr(self, inputs, test=False, sparse=False):

        self.image_lbs = inputs[0]
        self.image_ubs = inputs[1]
        self.image_label = inputs[2]
        self.unsafe_domains = inputs[3]
        self.input_shape = self.image_lbs.shape
        vzono_set = Vzono(self.image_lbs, self.image_ubs, test=test, sparse=sparse)
        if self.is_cuda:
            vzono_set.to_cuda()
        flatten_post = False
        for self._layer in range(self.layer_num):
            type_name = type(self.sequential[self._layer]).__name__
            # ['Conv2d', 'Linear', 'ReLU', 'Flatten']
            if type_name == 'SparseConv2d' or type_name =='Conv2d':
                t0 = time.time()
                vzono_set = self.conv2d(vzono_set)
                print('Conv2d time: ', time.time() - t0)
            elif type_name == 'ReLU':
                t0 = time.time()
                if not flatten_post:
                    vzono_set = self.relu_over_appr(vzono_set)
                else: # fully connected layers
                    vzono_set = self.relu_over_appr2(vzono_set)
                print('ReLU time: ', time.time() - t0)
            elif type_name == 'Linear':
                t0 = time.time()
                vzono_set = self.linear(vzono_set)
                print('Linear time: ', time.time() - t0)
            elif type_name == 'Flatten':
                vzono_set = self.flatten(vzono_set)
                flatten_post = True
            else:
                sys.exit('This layer type is not supported yet!')

            # self.layer_input_size.append(list(vzono_set.base_vectors.shape[1:]))

        # return self.verify_vzono(vzono_set)
        result, _ = self.verify_vzono(vzono_set)
        layer_data_dict = {'layer_old_inputs': self.layer_old_inputs, 'layer_neurons': self.layer_neurons, 'layer_vectors_sum': self.layer_vectors_sum}
        return result, layer_data_dict
        # return vzono_set


    def verify_vzono(self, vzono_set):
        vals = torch.sum(torch.abs(vzono_set.base_vectors), dim=0)
        ubs = vzono_set.base_vertices + vals
        lbs = vzono_set.base_vertices - vals

        # # print('ubs - lbs: ', ubs-lbs)
        # # print()
        #
        # if len(torch.nonzero(ubs >= lbs[self.image_label])) >= 2:
        #     return False, [lbs, ubs] # unknown
        # else:
        #     print('Safe subset!')
        #     return True, [lbs, ubs] # safe
        values = []
        for ud in self.unsafe_domains:
            A = ud[0]
            d = ud[1]
            base_vertices = np.dot(A, vzono_set.base_vertices.T) + d
            base_vectors = np.dot(A, vzono_set.base_vectors.T)
            val = base_vertices - np.sum(np.abs(base_vectors), axis=1)
            values.append(val[0])

        min_val, id = torch.min(torch.tensor(values),dim=0)

        _, impact_inputs = torch.sort(torch.abs(vzono_set.base_vectors[:self.input_shape[1]*self.input_shape[2]*self.input_shape[3],id[0]]), descending=True, dim=0)
        self.impact_inputs = impact_inputs.tolist()
        if min_val>=0:
            return True, [lbs, ubs] # safe
        else:
            return False, [lbs, ubs] # unknown


    def compute_impact_dim(self, dim, w, layer): # impactful dim in the preceding layer, conv2d
        if layer == -1:
            return dim, w

        conv2d = self.sequential[layer-1]
        kernel_size = conv2d.kernel_size
        assert kernel_size == (3,3)
        stride = conv2d.stride
        assert  stride == (1,1)
        padding = conv2d.padding
        weights = conv2d.weight.data

        weight_block = cp.deepcopy(weights[dim[0]])
        if padding[0] == 0:
            dim[1] += 1
            dim[2] += 1
        while True:
            index_raw = torch.argmax(torch.abs(weight_block))
            index_in_block = torch.tensor([index_raw//(3*3), index_raw%(3*3)//3, index_raw%(3*3)%3])
            value = weight_block[index_in_block[0], index_in_block[1], index_in_block[2]]
            index_in_input = cp.copy(index_in_block)
            index_in_input[1] = index_in_input[1] + dim[1] - 1
            index_in_input[2] = index_in_input[2] + dim[2] - 1
            if (index_in_input[1] < 0) or (index_in_input[1] > self.layer_input_size[layer][1]-1) or \
                    (index_in_input[2] < 0) or (index_in_input[2] > self.layer_input_size[layer][1]-1):
                weight_block[index_in_block[0], index_in_block[1], index_in_block[2]] = 0
                continue
            else:
                break

        w = w * value
        return self.compute_impact_dim(index_in_input, w, layer-2)


    def split_input(self, inputs_list, num=4):
        split_dims = []
        for _ in range(num):
            inputs_list, dim, input_dim = self.split_input_once2(inputs_list)
            split_dims.append([dim, input_dim])
        return inputs_list, split_dims


    def split_input_once2(self, inputs_list):
        s = list(self.image_lbs.shape)
        dim = self.impact_inputs.pop(0)
        assert len(s) == 4
        input_dim = [dim//(s[2]*s[3]), dim%(s[2]*s[3])//s[3], dim%(s[2]*s[3])%s[3]]

        new_inputs = []
        for inputs in inputs_list:
            image_lbs = inputs[0]
            image_ubs = inputs[1]
            image_lbs0 = cp.deepcopy(image_lbs)
            image_ubs0 = cp.deepcopy(image_ubs)
            image_lbs1 = cp.deepcopy(image_lbs)
            image_ubs1 = cp.deepcopy(image_ubs)

            dim_lb = image_lbs[0, input_dim[0], input_dim[1], input_dim[2]]
            dim_ub = image_ubs[0, input_dim[0], input_dim[1], input_dim[2]]

            ratio = 1/2
            split_point = ratio * (dim_ub - dim_lb) + dim_lb

            image_ubs0[0, input_dim[0], input_dim[1], input_dim[2]] = split_point
            image_lbs1[0, input_dim[0], input_dim[1], input_dim[2]] = split_point

            new_inputs.append([image_lbs0, image_ubs0, self.image_label, self.unsafe_domains])
            new_inputs.append([image_lbs1, image_ubs1, self.image_label, self.unsafe_domains])

        return new_inputs, dim, input_dim


    def split_input_once(self, inputs_list):
        for layer, items in enumerate(self.relus_over):
            if items:
                break
        dim = items[0].pop(0)
        lb_split = items[1].pop(0)
        ub_split = items[2].pop(0)
        weight = 1.0
        input_dim, w = self.compute_impact_dim(dim, weight, layer)
        assert lb_split < 0 and ub_split > 0

        ratio = -lb_split/(-lb_split+ub_split)
        # print('ratio: ', ratio)

        new_inputs = []
        for inputs in inputs_list:
            image_lbs = inputs[0]
            image_ubs = inputs[1]
            image_lbs0 = cp.deepcopy(image_lbs)
            image_ubs0 = cp.deepcopy(image_ubs)
            image_lbs1 = cp.deepcopy(image_lbs)
            image_ubs1 = cp.deepcopy(image_ubs)

            dim_lb = image_lbs[0, input_dim[0], input_dim[1], input_dim[2]]
            dim_ub = image_ubs[0, input_dim[0], input_dim[1], input_dim[2]]

            # ratio = 1/2
            if w > 0:
                split_point = ratio * (dim_ub-dim_lb) + dim_lb
            else:
                split_point = (1-ratio) * (dim_ub-dim_lb) + dim_lb

            image_ubs0[0, input_dim[0], input_dim[1], input_dim[2]] = split_point
            image_lbs1[0, input_dim[0], input_dim[1], input_dim[2]] = split_point

            new_inputs.append([image_lbs0, image_ubs0, self.image_label, self.unsafe_domains])
            new_inputs.append([image_lbs1, image_ubs1, self.image_label, self.unsafe_domains])

        return new_inputs



    def conv2d(self, vzono_set):
        vzono_set.base_vertices = self.sequential[self._layer](vzono_set.base_vertices)
        bias = self.sequential[self._layer].bias.data
        conv2d_nobias = cp.deepcopy(self.sequential[self._layer])

        conv2d_nobias.bias = torch.nn.Parameter(torch.zeros(bias.shape),requires_grad=False)
        vzono_set.base_vectors = conv2d_nobias(vzono_set.base_vectors)
        return vzono_set


    def conv2d_post(self, vzonop_set):
        vzonop_set.base_vertices = self.sequential[self._layer](vzonop_set.base_vertices)
        bias = self.sequential[self._layer].bias.data
        conv2d_nobias = cp.deepcopy(self.sequential[self._layer])

        conv2d_nobias.bias = torch.nn.Parameter(torch.zeros(bias.shape),requires_grad=False)
        vzonop_set.base_vector_bias = conv2d_nobias(vzonop_set.base_vector_bias)
        return vzonop_set


    def linear(self, vzono_set):
        vzono_set.base_vertices = self.sequential[self._layer](vzono_set.base_vertices)
        bias = self.sequential[self._layer].bias.data
        linear_nobias = cp.deepcopy(self.sequential[self._layer])
        linear_nobias.bias = torch.nn.Parameter(torch.zeros(bias.shape), requires_grad=False)
        vzono_set.base_vectors = linear_nobias(vzono_set.base_vectors)
        return vzono_set


    def linear_post(self, vzonop_set):
        vzonop_set.base_vertices = self.sequential[self._layer](vzonop_set.base_vertices)
        bias = self.sequential[self._layer].bias.data
        linear_nobias = cp.deepcopy(self.sequential[self._layer])
        linear_nobias.bias = torch.nn.Parameter(torch.zeros(bias.shape), requires_grad=False)
        vzonop_set.base_vector_bias = linear_nobias(vzonop_set.base_vector_bias)
        return vzonop_set


    def flatten(self, vzono_set):
        vzono_set.base_vertices = torch.flatten(vzono_set.base_vertices, start_dim=1)
        vzono_set.base_vectors = torch.reshape(vzono_set.base_vectors, (vzono_set.base_vectors.shape[0],-1))
        # vzono_set.base_vectors = vzono_set.base_vectors.to_sparse()
        return  vzono_set


    def flatten_post(self, vzonop_set):
        vzonop_set.base_vertices = torch.flatten(vzonop_set.base_vertices, start_dim=1)
        vzonop_set.base_vector_bias = torch.reshape(vzonop_set.base_vector_bias, (vzonop_set.base_vector_bias.shape[0],-1))
        # vzono_set.base_vectors = vzono_set.base_vectors.to_sparse()
        return  vzonop_set


    def relu_over_appr_post(self, vzonop_set):
        old_base_vertices = vzonop_set.layer_data['layer_old_inputs'][self._layer]
        old_neurons_over = vzonop_set.layer_data['layer_neurons'][self._layer]
        old_vectors_sum = vzonop_set.layer_data['layer_vectors_sum'][self._layer]

        base_vector_bias = vzonop_set.base_vector_bias
        new_base_vertices = vzonop_set.base_vertices

        new_vectors_sum = old_vectors_sum - torch.abs(base_vector_bias)
        vals_min = (new_base_vertices - new_vectors_sum)[0, old_neurons_over[:,0],old_neurons_over[:,1],old_neurons_over[:,2]]
        vals_max = (new_base_vertices + new_vectors_sum)[0, old_neurons_over[:,0],old_neurons_over[:,1],old_neurons_over[:,2]]


        neurons_neg = torch.any(vals_max<=0, dim=0)
        new_base_vertices[:, neurons_neg] = 0.0
        new_vectors_sum[:, neurons_neg] = 0.0
        valid_neurons_pos = torch.any(vals_min>=0, dim=0)
        valid_neurons_neg_pos = ~(valid_neurons_neg + valid_neurons_pos)





        valid_neurons_neg = torch.all((vzonop_set.base_vertices + vals) <= 0, dim=0)
        valid_neurons_pos = torch.all((vzonop_set.base_vertices - vals) >= 0, dim=0)
        valid_neurons_neg_pos = ~(valid_neurons_neg + valid_neurons_pos)

        xx = 1





    def relu_over_appr(self, vzono_set):
        if self.sparse:
            vzono_set.base_vertices = vzono_set.base_vertices.dense()
            vzono_set.base_vectors = vzono_set.base_vectors.dense()

        neurons_neg_pos, neurons_neg, vals = self.get_valid_neurons_for_over_app(vzono_set)
        # t0 = time.time()
        vzono_set.base_vertices[:, neurons_neg] = 0
        vzono_set.base_vectors[:, neurons_neg] = 0
        # print('relu_over_appr time: ', time.time() - t0)

        if not torch.any(neurons_neg_pos):
            return vzono_set

        # base_vectices = vzono_set.base_vertices[:, neurons_neg_pos]
        # base_vectices_max, _ = torch.max(vzono_set.base_vertices[:, neurons_neg_pos], dim=0, keepdim=True)
        # base_vectices_min, _ = torch.min(vzono_set.base_vertices[:, neurons_neg_pos], dim=0, keepdim=True)
        # ubs = base_vectices_max + vals[:, neurons_neg_pos]
        # lbs = base_vectices_min - vals[:, neurons_neg_pos]
        ubs = vzono_set.base_vertices[:, neurons_neg_pos] + vals[:, neurons_neg_pos]
        lbs = vzono_set.base_vertices[:, neurons_neg_pos] - vals[:, neurons_neg_pos]
        M = ubs / (ubs - lbs)
        epsilons = -lbs * M / 2

        # if self.collect_layer_flag:
        #     _, indices = torch.sort(-ubs*lbs, descending=True, dim=1)
        #     neurons_list = torch.nonzero(neurons_neg_pos)[indices].tolist()[0]
        #     lbs_list = lbs[0,:][indices].tolist()[0]
        #     ubs_list = ubs[0,:][indices].tolist()[0]
        #     self.relus_over[self._layer] = [neurons_list, lbs_list, ubs_list]
        # self.layer_neurons[self._layer] = torch.nonzero(neurons_neg_pos)
        # self.layer_vectors_sum[self._layer] = vals
        # self.layer_old_inputs[self._layer] = cp.deepcopy(vzono_set.base_vertices)

        vzono_set.base_vertices[:, neurons_neg_pos] = vzono_set.base_vertices[:, neurons_neg_pos] * M + epsilons
        vzono_set.base_vectors[:, neurons_neg_pos] = vzono_set.base_vectors[:, neurons_neg_pos] * M

        neurons_neg_pos_index = torch.nonzero(neurons_neg_pos)
        new_base_vectors_shape = [neurons_neg_pos_index.shape[0]]
        new_base_vectors_shape.extend(list(neurons_neg_pos.size()))

        if self.is_cuda:
            new_base_vectors = torch.zeros(tuple(new_base_vectors_shape)).cuda()
        else:
            new_base_vectors = torch.zeros(tuple(new_base_vectors_shape))

        n = 0
        for index in neurons_neg_pos_index:
            new_base_vectors[n, index[0],index[1],index[2]] = epsilons[0,n]
            n += 1

        vzono_set.base_vectors = torch.cat((vzono_set.base_vectors, new_base_vectors), dim=0)
        if self.sparse:
            vzono_set.base_vectors = spconv.SparseConvTensor.from_dense(vzono_set.base_vectors.permute(0, 2, 3, 1))
            vzono_set.base_vertices = spconv.SparseConvTensor.from_dense(vzono_set.base_vertices.permute(0, 2, 3, 1))
        return vzono_set


    def relu_over_appr2(self, vzono_set):
        neurons_neg_pos, neurons_neg, vals = self.get_valid_neurons_for_over_app2(vzono_set)
        vzono_set.base_vertices[:,neurons_neg] = 0
        vzono_set.base_vectors[:,neurons_neg] = 0

        if not torch.any(neurons_neg_pos):
            return vzono_set

        ubs = vzono_set.base_vertices[:, neurons_neg_pos] + vals[:, neurons_neg_pos]
        lbs = vzono_set.base_vertices[:, neurons_neg_pos] - vals[:, neurons_neg_pos]
        M = ubs / (ubs - lbs)
        epsilons = -lbs * M / 2

        vzono_set.base_vertices[:, neurons_neg_pos] = vzono_set.base_vertices[:, neurons_neg_pos] * M + epsilons
        vzono_set.base_vectors[:, neurons_neg_pos] = vzono_set.base_vectors[:, neurons_neg_pos] * M

        neurons_neg_pos_index = torch.nonzero(neurons_neg_pos)
        new_base_vectors_shape = [len(neurons_neg_pos_index), neurons_neg_pos.shape[0]]
        if self.is_cuda:
            new_base_vectors = torch.zeros(tuple(new_base_vectors_shape)).cuda()
        else:
            new_base_vectors = torch.zeros(tuple(new_base_vectors_shape))
        new_base_vectors[range(new_base_vectors_shape[0]), neurons_neg_pos_index[:,0]] = epsilons

        vzono_set.base_vectors = torch.cat((vzono_set.base_vectors, new_base_vectors), dim=0)
        return vzono_set



    def get_valid_neurons_for_over_app(self, vfl_set):
        vals = torch.sum(torch.abs(vfl_set.base_vectors), dim=0, keepdim=True)
        valid_neurons_neg = torch.all((vfl_set.base_vertices+vals)<=0, dim=0)
        valid_neurons_pos = torch.all((vfl_set.base_vertices-vals)>=0, dim=0)
        valid_neurons_neg_pos = ~(valid_neurons_neg + valid_neurons_pos)
        return valid_neurons_neg_pos, valid_neurons_neg, vals


    def get_valid_neurons_for_over_app_sparse(self, base_vertices, base_vectors):
        vals = torch.sum(torch.abs(base_vectors), dim=0, keepdim=True)
        valid_neurons_neg = torch.all((base_vertices+vals)<=0, dim=0)
        valid_neurons_pos = torch.all((base_vertices-vals)>=0, dim=0)
        valid_neurons_neg_pos = ~(valid_neurons_neg + valid_neurons_pos)
        return valid_neurons_neg_pos, valid_neurons_neg, vals


    def get_valid_neurons_for_over_app2(self, vfl_set):
        vals = torch.sum(torch.abs(vfl_set.base_vectors), dim=0, keepdim=True)
        valid_neurons_neg = torch.all((vfl_set.base_vertices+vals)<=0, dim=0)
        valid_neurons_pos = torch.all((vfl_set.base_vertices-vals)>=0, dim=0)
        valid_neurons_neg_pos = ~(valid_neurons_neg + valid_neurons_pos)
        return valid_neurons_neg_pos, valid_neurons_neg, vals







