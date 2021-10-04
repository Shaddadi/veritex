import numpy as np
import copy as cp
from vzono_cnn import Vzono
import sys
import torch
import time


class CNN:

    def __init__(self, sequential, is_cuda=False):
        if is_cuda:
            self.sequential = sequential.eval().cuda()
        else:
            self.sequential = sequential.eval()

        self.is_cuda = is_cuda
        self.layer_num = len(sequential)
        self.relus_over = [None] * (self.layer_num) # over-approximated relus for future splitting
        self.collect_layer_flag = True
        self.layer_input_size = []
        self._layer = None


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

        if not inputs_list:
            return True # safe
        else:
            return False # unknown




    def reach_over_appr(self, inputs):

        self.image_lbs = inputs[0]
        self.image_ubs = inputs[1]
        self.image_label = inputs[2]
        vzono_set = Vzono(self.image_lbs, self.image_ubs)
        if self.is_cuda:
            vzono_set.to_cuda()

        for self._layer in range(self.layer_num):
            type_name = type(self.sequential[self._layer]).__name__
            # ['Conv2d', 'Linear', 'ReLU', 'Flatten']
            if type_name == 'Conv2d':
                vzono_set = self.conv2d(vzono_set)
            elif type_name == 'ReLU':
                if len(vzono_set.base_vectors.shape) == 4:
                    vzono_set = self.relu_over_appr(vzono_set)
                elif len(vzono_set.base_vectors.shape) == 2: # fully connected layers
                    vzono_set = self.relu_over_appr2(vzono_set)
            elif type_name == 'Linear':
                vzono_set = self.linear(vzono_set)
            elif type_name == 'Flatten':
                vzono_set = self.flatten(vzono_set)
            else:
                sys.exit('This layer type is not supported yet!')

            self.layer_input_size.append(list(vzono_set.base_vectors.shape[1:]))

        return self.verify_vzono(vzono_set)


    def verify_vzono(self, vzono_set):
        vals = torch.sum(torch.abs(vzono_set.base_vectors), dim=0)
        ubs = vzono_set.base_vertices + vals
        lbs = vzono_set.base_vertices - vals
        # print('ubs - lbs: ', ubs-lbs)
        # print()

        if len(torch.nonzero(ubs >= lbs[self.image_label])) >= 2:
            return False, [lbs, ubs] # unknown
        else:
            print('Safe subset!')
            return True, [lbs, ubs] # safe


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


    def split_input(self, inputs_list, num=2):
        for _ in range(num):
            inputs_list = self.split_input_once(inputs_list)
        return inputs_list


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

            new_inputs.append([image_lbs0, image_ubs0, self.image_label])
            new_inputs.append([image_lbs1, image_ubs1, self.image_label])

        return new_inputs





    def conv2d(self, vzono_set):
        vzono_set.base_vertices = self.sequential[self._layer](vzono_set.base_vertices)
        bias = self.sequential[self._layer].bias.data
        bias = torch.reshape(bias,(1,len(bias),1,1))
        vzono_set.base_vectors = self.sequential[self._layer](vzono_set.base_vectors) - bias

        return vzono_set


    def linear(self, vzono_set):
        vzono_set.base_vertices = self.sequential[self._layer](vzono_set.base_vertices)
        bias = self.sequential[self._layer].bias.data
        bias = torch.reshape(bias, (1, len(bias)))
        vzono_set.base_vectors = self.sequential[self._layer](vzono_set.base_vectors) - bias
        return vzono_set


    def flatten(self, vzono_set):
        vzono_set.base_vertices = torch.flatten(vzono_set.base_vertices)
        vzono_set.base_vectors = torch.reshape(vzono_set.base_vectors, (vzono_set.base_vectors.shape[0],-1))
        vzono_set.base_vectors = vzono_set.base_vectors.to_sparse()
        return  vzono_set


    def relu_over_appr(self, vzono_set):
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

        if self.collect_layer_flag:
            _, indices = torch.sort(-ubs*lbs, descending=True, dim=1)
            neurons_list = torch.nonzero(neurons_neg_pos)[indices].tolist()[0]
            lbs_list = lbs[0,:][indices].tolist()[0]
            ubs_list = ubs[0,:][indices].tolist()[0]
            self.relus_over[self._layer] = [neurons_list, lbs_list, ubs_list]

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
        return vzono_set


    def relu_over_appr2(self, vzono_set):
        neurons_neg_pos, neurons_neg, vals = self.get_valid_neurons_for_over_app2(vzono_set)
        vzono_set.base_vertices[neurons_neg] = 0
        vzono_set.base_vectors[:,neurons_neg] = 0

        if not torch.any(neurons_neg_pos):
            return vzono_set

        # base_vectices = vzono_set.base_vertices[:, neurons_neg_pos]
        base_vectices_max, _ = torch.max(vzono_set.base_vertices[neurons_neg_pos], dim=0, keepdim=True)
        base_vectices_min, _ = torch.min(vzono_set.base_vertices[neurons_neg_pos], dim=0, keepdim=True)

        ubs = base_vectices_max + vals[:, neurons_neg_pos]
        lbs = base_vectices_min - vals[:, neurons_neg_pos]
        M = ubs / (ubs - lbs)
        epsilons = -lbs * M / 2

        vzono_set.base_vertices[neurons_neg_pos] = vzono_set.base_vertices[neurons_neg_pos] * M + epsilons
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


    def get_valid_neurons_for_over_app2(self, vfl_set):
        vals = torch.sum(torch.abs(vfl_set.base_vectors), dim=0, keepdim=True)
        valid_neurons_neg = torch.all((vfl_set.base_vertices+vals)<=0, dim=0)
        valid_neurons_pos = torch.all((vfl_set.base_vertices-vals)>=0, dim=0)
        valid_neurons_neg_pos = ~(valid_neurons_neg + valid_neurons_pos)
        return valid_neurons_neg_pos, valid_neurons_neg, vals







