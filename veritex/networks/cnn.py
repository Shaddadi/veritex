import numpy as np
import itertools
import copy as cp
import torch
import torch.nn as nn
import torch.multiprocessing
from torch.autograd import Variable
import matplotlib.pyplot as plt
import cubelattice as cl
import scipy.io as sio
import inconfig
import pickle
import time
import sys


class Flatten(nn.Module):
    def forward(self, input):
        return torch.flatten(input,1)

class network:
    def __init__(self, net, image, label, attack_pos, layer_gradients,
                 is_cuda=False, top_dims=1):

        self.pool = None
        self.class_label = None
        self.flatten_pos = None # the position of the flatten function in layers
        self.is_cuda = is_cuda
        self.layer_gradients = layer_gradients
        self.top_dims = top_dims # top_dims<1 indicates under approximation
        self.label = label
        self.start_time = 0

        for param in net.parameters():
            param.requires_grad = False

        self.sequential = self.forward_layer_sequential(net)
        self.layer_num = len(self.sequential) # the number of layers
        self.all_inputs = self.forward_layer_input(image)

        self.all_range_pos, self.all_attack_pos = self.layers_range(attack_pos)
        if self.top_dims < 1:
            self.indices_selected_layers = self.get_target_indices()

        self._layer = None
        self._image_frame = None
        self._slice_blocks = None


    def forward_layer_sequential(self, net):
        net_layers = []
        for i in range(len(net.sequential)):
            type_name = type(net.sequential[i]).__name__
            if type_name in ['Flatten', 'MaxPool2d', 'BatchNorm2d','ReLU','DataParallel','Linear','Conv2d']:
                net_layers.append(net.sequential[i])
            if type_name == 'Flatten':
                self.flatten_pos = i

        if self.is_cuda:
            sequential_layers = nn.Sequential(*net_layers).eval().cuda()
        else:
            sequential_layers = nn.Sequential(*net_layers).eval()

        for param in sequential_layers.parameters():
            param.requires_grad = False

        return sequential_layers


    def forward_layer_input(self, image):
        if self.is_cuda:
            im = copy.deepcopy(image).cuda()
            all_inputs = [im.cpu().numpy()]
            for i in range(len(self.sequential)):
                im = self.sequential[i](im)
                all_inputs.append(copy.deepcopy(im.cpu().numpy()))

        else:
            im = copy.deepcopy(image)
            all_inputs = [im.numpy()]
            for i in range(len(self.sequential)):
                im = self.sequential[i](im)
                all_inputs.append(copy.deepcopy(im.numpy()))

        self.class_label = np.argmax(all_inputs[-1],1)

        return all_inputs


    def regular_reach(self, initial_image_fl):

        layer_fls = [initial_image_fl]
        for self._layer in range(self.layer_num):
            layer_name = type(self.sequential[self._layer]).__name__
            print('***************************************')
            print('Layer type: ', layer_name)
            t0 = time.time()

            self.reach_single_layer(layer_fls)

            print('Time: ', time.time()-t0)
            print('Layer: ', self._layer, '/', self.layer_num)
            print('Number of sets: ', len(layer_fls))
            print('\n')

        layer_fls_new = []
        for afl in layer_fls:
            temp = []
            temp.append(afl.vertices)
            temp.append(afl.vertices_init)
            layer_fls_new.append(temp)

        return layer_fls_new


    def index_convesion(self, indices, matrix_size):
        rawmaxidx = indices
        idx = []
        for adim in list(matrix_size)[::-1]:
            idx.append((rawmaxidx % adim).numpy())
            rawmaxidx = rawmaxidx / adim

        idx = np.asarray(idx[::-1]).transpose()
        return idx


    def get_target_indices(self):
        indices_selected_layers = []
        indices_non_selected_layers = []
        for layer in range(self.layer_num-1): # the last layer is not considered
            layer_name = type(self.sequential[layer]).__name__
            if layer_name == 'Flatten':
                indices_new = []
                indices_selected_layers.append(indices_new)
                indices_non_selected_layers.append(indices_new)
                continue

            gradients = self.layer_gradients[layer]
            if layer < self.flatten_pos:
                pos = self.all_attack_pos[layer]
                grads_pos = gradients[:, pos[0][0]:pos[1][0] + 1, pos[0][1]:pos[1][1] + 1]
                tops = int(np.ceil(self.top_dims*grads_pos.shape[0]*grads_pos.shape[1]*grads_pos.shape[2]))
                grads_pos_faltten = grads_pos.flatten()
                _, indices_topk = torch.topk(grads_pos_faltten, tops)

                indices_new_topk = self.index_convesion(indices_topk, grads_pos.size())

                if layer_name == 'MaxPool2d':
                    range_pos = self.all_range_pos[layer]
                    rel_pos = pos[0]-range_pos[0]
                    indices_new_topk[:,1] += rel_pos[0]
                    indices_new_topk[:,2] += rel_pos[1]
                    indices_new_topk = [tuple(arr) for arr in indices_new_topk]
                else:
                    indices_new_topk = [tuple(arr) for arr in indices_new_topk]

            else:
                grads_pos = gradients
                tops = int(np.ceil(self.top_dims *grads_pos.shape[0]))
                _, indices_topk = torch.topk(grads_pos, tops)
                indices_new_topk = indices_topk.tolist()

            indices_selected_layers.append(indices_new_topk)

        return indices_selected_layers


    def layers_range(self, initial_attack_pos):
        attack_pos = initial_attack_pos
        all_range_pos = []
        all_attack_pos = []
        for self._layer in range(self.flatten_pos):
            if type(self.sequential[self._layer]).__name__ == 'ReLU' or \
                    type(self.sequential[self._layer]).__name__ == 'BatchNorm2d':
                all_attack_pos.append(attack_pos)
                all_range_pos.append(attack_pos)
                continue

            all_attack_pos.append(attack_pos)
            range_to_process, next_attack_pos = self.range_to_process_layer(attack_pos)
            all_range_pos.append(range_to_process)
            attack_pos = next_attack_pos

        all_attack_pos.append(attack_pos) # add the attacked positions in the flatten() layer

        return all_range_pos, all_attack_pos


    def reach_single_layer(self, all_fls):
        if type(self.sequential[self._layer]).__name__ == 'DataParallel':
            type_name = type(self.sequential[self._layer].module).__name__
        else:
            type_name = type(self.sequential[self._layer]).__name__

        if type_name == 'BatchNorm2d':
            self.norm2d_layer(all_fls)

        elif type_name == 'Conv2d':
            rp = self.all_range_pos[self._layer]
            self._image_frame = self.all_inputs[self._layer][:, :, rp[0][0]:rp[1][0] + 1, rp[0][1]:rp[1][1] + 1]

            self.conv2d_layer(all_fls)

        elif type_name == 'ReLU' and self._layer<self.flatten_pos:
            neurons = np.array([])
            all_fls_len = len(all_fls)
            num = 0
            while num < all_fls_len:
                im_fl = all_fls[0]
                all_fls.pop(0)
                try: #avoid numerical error
                    im_fl_output = self.relu_layer1(im_fl, neurons, False)
                    all_fls.extend(im_fl_output)
                except:
                    pass
                num += 1

        elif type_name == 'MaxPool2d':
            r2p = self.all_range_pos[self._layer]
            attack_pos = self.all_attack_pos[self._layer]
            pos = attack_pos - r2p[0]
            self._slice_blocks = self.get_slice_blocks(r2p)
            self._image_frame = self.all_inputs[self._layer][:, :, r2p[0][0]:r2p[1][0] + 1, r2p[0][1]:r2p[1][1] + 1]

            all_fls_length = len(all_fls)
            num = 0
            while num < all_fls_length:
                im_fl = all_fls[0]
                all_fls.pop(0)
                data_frame = self._image_frame.repeat(im_fl.vertices.shape[0], 0)
                data_frame[:, :, pos[0][0]:pos[1][0] + 1, pos[0][1]:pos[1][1] + 1] = cp.deepcopy(im_fl.vertices)
                im_fl.vertices = data_frame
                blocks = np.array([])
                try: #avoid numerical errors
                    im_fl_output = self.maxpool2d_layer(im_fl, blocks, False)
                    all_fls.extend(im_fl_output)
                except:
                    pass
                num += 1

        elif type_name == 'Flatten':
            self._image_frame = self.all_inputs[self._layer]

            for im_fl in all_fls:
                self.flatten_layer(im_fl)

        elif type_name == 'ReLU' and self._layer > self.flatten_pos:
            neurons = np.array([])
            all_fls_length = len(all_fls)
            num = 0
            while num < all_fls_length:
                im_fl = all_fls[0]
                all_fls.pop(0)
                try:
                    im_fl_output = self.relu_layer2(im_fl, neurons, False)
                    all_fls.extend(im_fl_output)
                except:
                    pass
                num += 1

        elif type_name == 'Linear':
            for im_fl in all_fls:
                self.linear_layer(im_fl)


    def norm2d_layer(self, all_fls):
        if self.is_cuda:
            for im_fl in all_fls:
                vertices = cp.deepcopy(torch.from_numpy(im_fl.vertices).cuda())
                vertices_norm = self.sequential[self._layer](vertices)
                im_fl.vertices = vertices_norm.cpu().numpy()
        else:
            for im_fl in all_fls:
                vertices = torch.from_numpy(im_fl.vertices)
                vertices_norm = self.sequential[self._layer](vertices)
                im_fl.vertices = vertices_norm.numpy()


    def conv2d_layer(self, all_fls):
        if self.is_cuda:
            ap = self.all_attack_pos[self._layer]
            ap_next = self.all_attack_pos[self._layer+1]
            rp = self.all_range_pos[self._layer]
            for im_fl in all_fls:
                im_input = cp.deepcopy(self._image_frame).repeat(im_fl.vertices.shape[0], axis=0)
                ap2 = ap - rp[0]
                im_input[:, :, ap2[0][0]:ap2[1][0] + 1, ap2[0][1]:ap2[1][1] + 1] = cp.deepcopy(im_fl.vertices)
                # while self.check_gpu_and_block(im_input): {}

                im_input0 = torch.from_numpy(im_input).cuda()
                im_output0 = self.sequential[self._layer](im_input0)
                im_output = im_output0.cpu().numpy()

                torch.cuda.empty_cache()
                ap3 = ap_next - rp[0]
                im_fl.vertices = cp.deepcopy(im_output[:, :, ap3[0][0]:ap3[1][0] + 1, ap3[0][1]:ap3[1][1] + 1])
        else:
            ap = self.all_attack_pos[self._layer]
            ap_next = self.all_attack_pos[self._layer + 1]
            rp = self.all_range_pos[self._layer]
            for im_fl in all_fls:
                im_input = cp.deepcopy(self._image_frame).repeat(im_fl.vertices.shape[0], axis=0)
                ap2 = ap - rp[0]
                im_input[:, :, ap2[0][0]:ap2[1][0] + 1, ap2[0][1]:ap2[1][1] + 1] = im_fl.vertices

                im_input0 = torch.from_numpy(im_input)
                im_output0 = self.sequential[self._layer](im_input0)
                im_output = im_output0.numpy()

                ap3 = ap_next - rp[0]
                im_fl.vertices = cp.deepcopy(im_output[:, :, ap3[0][0]:ap3[1][0] + 1, ap3[0][1]:ap3[1][1] + 1])


    def get_valid_neurons1(self, afl, neurons):
        if neurons.shape[0] ==0:
            flag_neg = (afl.vertices<=0)
            temp_neg = np.all(flag_neg, 0)
            valid_neurons_neg = np.asarray(np.nonzero(temp_neg)).T
            temp_pos = np.all(afl.vertices>=0, 0)
            neurons_sum = temp_neg+temp_pos
            valid_neurons_neg_pos = np.asarray(np.nonzero(neurons_sum==False)).T
            return valid_neurons_neg_pos, valid_neurons_neg

        elements = afl.vertices[:, neurons[:,0], neurons[:,1], neurons[:,2]]
        flag_neg = (elements <= 0)
        temp_neg = np.all(flag_neg, 0)
        temp_pos = np.all(elements>=0, 0)
        temp_sum = temp_neg + temp_pos
        indx_neg_pos = np.asarray(np.nonzero(temp_sum == False)).T
        valid_neurons_neg_pos = neurons[indx_neg_pos[:,0]]
        indx_neg = np.asarray(np.nonzero(temp_neg)).T
        valid_neurons_neg = neurons[indx_neg[:,0]]

        return valid_neurons_neg_pos, valid_neurons_neg

    def get_valid_neurons2(self, afl, neurons):
        if neurons.shape[0] ==0:
            flag_neg = (afl.vertices<=0)
            temp_neg = np.all(flag_neg, 0)
            valid_neurons_neg = np.asarray(np.nonzero(temp_neg)).T
            temp_pos = np.all(afl.vertices>=0, 0)
            neurons_sum = temp_neg + temp_pos
            valid_neurons_neg_pos = np.asarray(np.nonzero(neurons_sum==False))[0,:]
            return valid_neurons_neg_pos, valid_neurons_neg

        elements = afl.vertices[:,neurons[:,0]]
        flag_neg = (elements <= 0)
        temp_neg = np.all(flag_neg, 0)
        valid_neurons_neg = np.asarray(np.nonzero(temp_neg)).T
        temp_pos = np.all(elements>=0, 0)
        neurons_sum = temp_neg + temp_pos
        indx_neg_pos = np.asarray(np.nonzero(neurons_sum==False)).T
        valid_neurons_neg_pos = neurons[indx_neg_pos[:, 0]][0,:]
        return valid_neurons_neg_pos, valid_neurons_neg


    def relu_layer1(self, im_fl, neurons, flag=True):
        if (neurons.shape[0] == 0) and flag:
            return [im_fl]

        new_neurons, new_neurons_neg = self.get_valid_neurons1(im_fl, neurons)

        im_fl.map_negative_fl_multi_relu1(new_neurons_neg)

        if new_neurons.shape[0] == 0:
            return [im_fl]

        if self.top_dims<1 and (tuple(new_neurons[0]) not in self.indices_selected_layers[self._layer]):
            im_fl.fast_reach = True
        fls = self.split_facelattice(im_fl, new_neurons[0], 'relu')

        new_neurons = new_neurons[1:]

        all_fls = []
        for afl in fls:
            all_fls.extend(self.relu_layer1(afl, new_neurons))

        return all_fls


    def relu_layer2(self, im_fl, neurons, flag=True):
        if (neurons.shape[0] == 0) and flag:
            return [im_fl]

        new_neurons, new_neurons_neg = self.get_valid_neurons2(im_fl, neurons)
        im_fl.map_negative_fl_multi_relu2(new_neurons_neg)

        if new_neurons.shape[0] == 0:
            return [im_fl]

        if self.top_dims<1 and (new_neurons[0] not in self.indices_selected_layers[self._layer]):
            im_fl.fast_reach = True
        fls = self.split_facelattice(im_fl, new_neurons[0], 'relu2')
        new_neurons = new_neurons[1:]

        all_fls = []
        for afl in fls:
            all_fls.extend(self.relu_layer2(afl, new_neurons))

        return all_fls


    def linear_layer(self,im_fl):
        if self.is_cuda:
            im_input = im_fl.vertices
            # while self.check_gpu_and_block(im_input): pass
            temp0 = torch.from_numpy(im_input).cuda()

            temp1 = self.sequential[self._layer](temp0)
            im_fl.vertices = temp1.cpu().numpy()
            torch.cuda.empty_cache()

        else:
            temp0 = torch.from_numpy(im_fl.vertices)
            temp1 = self.sequential[self._layer](temp0)
            im_fl.vertices = temp1.numpy()


    def split_facelattice(self, im_fl, aneuron, split_type):
        if split_type=='relu':
            pos_fl, neg_fl = im_fl.single_split_relu1(aneuron)
        elif split_type=='maxpool':
            pos_fl, neg_fl = im_fl.single_split_maxpool(aneuron)
        elif split_type =='relu2':
            pos_fl, neg_fl = im_fl.single_split_relu2(aneuron)
        else:
            print('Split type is not defined!\n')

        outputs = []
        if pos_fl:
            outputs.append(pos_fl)
        if neg_fl:
            outputs.append(neg_fl)
        # return cp.deepcopy(outputs)
        return outputs


    def get_valid_blocks(self, blocks_tosplit, indices):
        if blocks_tosplit.shape[0]==0:
            flag_equ = (indices==indices[0,:,:,:])
            valid_blocks = np.asarray(np.nonzero(np.any(flag_equ==False, 0))).T
            return valid_blocks

        valid_indices = indices[:,blocks_tosplit[:,0],blocks_tosplit[:,1],blocks_tosplit[:,2]]
        flag_equal = (valid_indices==valid_indices[0,:])
        indx_temp = np.asarray(np.nonzero(np.any(flag_equal == False, 0))).T
        valid_blocks = blocks_tosplit[indx_temp[:,0]]
        return valid_blocks


    def maxpool2d_layer(self, image_fl, blocks_tosplit, flag=True):
        if self.is_cuda:
            self.sequential[self._layer].return_indices = True
            temp0 = torch.from_numpy(image_fl.vertices).cuda()
            layer_outs0, indices0 = self.sequential[self._layer](temp0)
            self.sequential[self._layer].return_indices = False
            layer_outs = layer_outs0.cpu().numpy()
            indices = indices0.cpu().numpy()
            torch.cuda.empty_cache()
        else:
            self.sequential[self._layer].return_indices = True
            temp0 = torch.from_numpy(image_fl.vertices)
            layer_outs0, indices0 = self.sequential[self._layer](temp0)
            self.sequential[self._layer].return_indices = False
            layer_outs = layer_outs0.numpy()
            indices = indices0.numpy()

        if blocks_tosplit.shape[0]==0 and flag:
            image_fl.vertices = layer_outs
            return [image_fl]

        blocks_tosplit_new = self.get_valid_blocks(blocks_tosplit, indices)

        if blocks_tosplit_new.shape[0]==0:
            image_fl.vertices = layer_outs
            return [image_fl]

        ablock = blocks_tosplit_new[0]
        blocks_tosplit_new = blocks_tosplit_new[1:]
        indices_flatten = np.unique(indices[:,ablock[0],ablock[1],ablock[2]])
        aset_elements = [[ablock[0]]+self._slice_blocks[idx] for idx in indices_flatten]
        hyperplanes = list(itertools.combinations(aset_elements, 2))

        fls_temp = [image_fl]
        for hp in hyperplanes:
            fls_temp_hp = []
            if self.top_dims<1 and (tuple(hp[0]) not in self.indices_selected_layers[self._layer]) and \
                    (tuple(hp[1]) not in self.indices_selected_layers[self._layer]):
                temp_fast = True
            else:
                temp_fast = False

            for afl in fls_temp:
                afl.fast_reach = temp_fast
                fls_temp_hp.extend(self.split_facelattice(afl, hp, 'maxpool'))

            fls_temp = fls_temp_hp

        all_fls = []
        for afl in fls_temp:
            all_fls.extend(self.maxpool2d_layer(afl, blocks_tosplit_new))

        return all_fls


    def get_slice_blocks(self, range_to_process):
        # single depth slice
        width = range_to_process[1][1]-range_to_process[0][1]+1
        height = range_to_process[1][0]-range_to_process[0][0]+1
        blocks = []
        for h in range(height):
            for w in range(width):
                ablock = [h, w]
                blocks.append(ablock)

        # blocks = torch.tensor(blocks)
        return blocks


    def flatten_layer(self, image_fl):
        ap = self.all_attack_pos[self._layer]
        data_frame = cp.deepcopy(self._image_frame).repeat(image_fl.vertices.shape[0], 0)
        data_frame[:, :, ap[0][0]:ap[1][0] + 1, ap[0][1]:ap[1][1] + 1] = cp.deepcopy(image_fl.vertices)
        image_fl.vertices = data_frame.reshape(image_fl.vertices.shape[0],-1)


    def range_to_process_layer(self, attack_pos):
        if type(self.sequential[self._layer]).__name__ == 'DataParallel':
            layer_function = self.sequential[self._layer].module
        else:
            layer_function = self.sequential[self._layer]

        kernel_size = np.array(layer_function.kernel_size)
        stride = np.array(layer_function.stride)
        padding = np.array(layer_function.padding)
        ub_pos = attack_pos[1]
        lb_pos = attack_pos[0]
        nmax = np.floor(np.divide(ub_pos + padding, stride)).astype(int)
        nmin = np.ceil(np.divide(lb_pos + padding + 1 - kernel_size, stride)).astype(int)
        nmin[nmin<0] = 0
        if nmax[0] > self.all_inputs[self._layer].shape[2] - 1:
            nmax[0] = self.all_inputs[self._layer].shape[2] - 1
        if nmax[1] > self.all_inputs[self._layer].shape[3] - 1:
            nmax[1] = self.all_inputs[self._layer].shape[3] - 1

        lb = nmin * stride - padding
        lb[lb < 0] = 0
        ub = nmax * stride + kernel_size - padding - 1
        if ub[0] > self.all_inputs[self._layer].shape[2] - 1:
            ub[0] = self.all_inputs[self._layer].shape[2] - 1
        if ub[1] > self.all_inputs[self._layer].shape[3] - 1:
            ub[1] = self.all_inputs[self._layer].shape[3] - 1

        range_to_process = [lb, ub]
        next_attack_pos = [nmin, nmax]
        return range_to_process, next_attack_pos

    def create_image_fl(self, image):
        clattice = cl.cubelattice()
        image_fl = clattice.to_facelattice()
        return image_fl

    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['pool']
        return self_dict

    def __setstate__(self, state):
        self.__dict__.update(state)


class Sequence():
    def __init__(self, my_list):
        self.my_list = my_list

    def __getitem__(self, item):
        return self.my_list[item]

    def __len__(self):
        return len(self.my_list)



class Hook():
    def __init__(self, module=None, backward=False):
        self.input = None
        self.output = None
        if module != None:
            if backward == False:
                self.hook = module.register_forward_hook(self.hook_fn)
            else:
                self.hook = module.register_backward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output

    def close(self):
        self.hook.remove()


class Method:
    def __init__(self, model, image, label, epsilon, file_path, pixel_block=[1,1],
                 mean=np.array([0,0,0]), std=np.array([1,1,1]),num_core=1, top_dims=1, target=None,  is_cuda=False):
        self.model = model
        self.image_orig = image[0]
        self.image= cp.deepcopy(image[0])
        self.label = label
        self.pixel_block = pixel_block
        self.attack_target = target
        self.pool = None
        self.num_core = num_core
        self.is_cuda = is_cuda
        self.epsilon = epsilon

        self.mean, self.std = mean, std
        self.attack_pos, self.layer_gradients = self.layer_gradients()
        
        self.poss = self.attack_pos[0]
        self.elapsed_time = 0
        self.savepath = file_path
        self.top_dims = top_dims
        if self.attack_target == self.label:
            sys.exit()


    def simulate(self, ablock):
        self.attack_range_3channel(self.poss)
        lbs = self.attack_range[0].flatten()
        ubs = self.attack_range[1].flatten()
        num = 100000

        rands = []
        for idx in range(len(lbs)):
            lb = lbs[idx]
            ub = ubs[idx]
            rands.append(np.random.uniform(lb, ub, num))

        rands = np.array(rands).transpose().reshape((num, 3, ablock[0], ablock[1]))

        image = self.image_orig.numpy()
        ap = self.poss
        outputs = np.array([])
        for n in range(num//100):
            sub = rands[n*100:(n+1)*100]
            image_frame = image.repeat(sub.shape[0], 0)
            image_frame[:, :, ap[0][0]: ap[1][0] + 1, ap[0][1]: ap[1][1] + 1] = sub
            image_frame = torch.tensor(image_frame)
            if n==0:
                outputs = self.model(image_frame.cuda()).cpu().numpy()
            else:
                outputs = np.concatenate((outputs, self.model(image_frame.cuda()).cpu().numpy()), axis=0)

        outputs = np.array(outputs)
        if self.model_name == 'vgg16':
            with open('./results/simulation.pkl', 'wb') as f:
                pickle.dump([outputs], f)
        else:
            with open('./results-distilled/simulation.pkl', 'wb') as f:
                pickle.dump([outputs], f)



    def reach(self):
        t0 = time.time()
        net = network(self.model, self.image, self.label, self.attack_pos[0],
                           self.layer_gradients, top_dims=self.top_dims, is_cuda=self.is_cuda)

        self.attack_range_3channel(self.poss)
        all_input_fls = inconfig.partition_input(self.attack_range,  pnum=4,
                                                 sp=self.poss)
        net.pool = torch.multiprocessing.Pool(self.num_core)  # multiprocessing

        outputSets = []
        outputSets.extend(net.pool.imap(net.regular_reach, all_input_fls))
        net.pool.close()
        self.elapsed_time = time.time() - t0
        all_fls = [item for sublist in outputSets for item in sublist]

        print('Neurons: '+str(self.top_dims*100)+'%')
        print('Epsilon: ', self.epsilon)
        print('Running Time: ', self.elapsed_time)
        print('Number of Output Sets: ', len(all_fls))
        with open(self.savepath+'.pkl', 'wb') as f:
            pickle.dump([all_fls, self.label.numpy(), self.label2.numpy(), self.elapsed_time, self.attack_pos[0], self.top_dims], f)

        sio.savemat(self.savepath+'.mat', {'all_fls': all_fls, 'label': self.label.numpy(), 'label2': self.label2.numpy()})



    def attack_range_3channel(self, pos):
        blocks = self.image[:,:,pos[0][0]:(pos[1][0]+1), pos[0][1]:(pos[1][1]+1)]

        for n in range(3):
            blocks[0,n,:,:] = blocks[0,n,:,:]*self.std[n]+self.mean[n]

        ub = cp.copy(blocks + self.epsilon)
        lb = cp.copy(blocks - self.epsilon)
        ub[ub>1.0] = 1.0
        lb[lb<0.0] = 0.0

        for n in range(3):
            ub[0,n,:,:] = (ub[0,n,:,:]-self.mean[n])/self.std[n]
            lb[0,n,:,:] = (lb[0,n,:,:]-self.mean[n])/self.std[n]

        self.attack_range=[lb[0].numpy(),ub[0].numpy()]



    def falsification_one_step(self):
        t0 = time.time()
        self.flag = True
        self.attack_pos_all = []
        self.adv_image = None
        self.adv_label = None
        pool = torch.multiprocessing.Pool(self.num_core)  # multiprocessing

        pos = self.attack_pos[0]

        net = network(cp.deepcopy(self.model), self.image, self.label, pos,
                           self.layer_gradients, top_dims=self.top_dims, is_cuda=self.is_cuda)
        net.pool = pool
        self.attack_range_3channel(pos)
        all_input_fls = inconfig.partition_input(self.attack_range, pnum=5,
                                                 sp=pos)

        outputSets = []
        outputSets.extend(net.pool.imap(net.regular_reach, all_input_fls))

        all_output_fls = [item for sublist in outputSets for item in sublist]

        self.elapsed_time = time.time() - t0
        print('Number of outputs: ', len(all_output_fls))
        print('Time: ', self.elapsed_time)

        self.attack_pos_all.append(pos)
        new_output = self.find_optimal_image_one_step(all_output_fls)

        pool.close()

        ori_output = self.model(self.image_orig)[0]
        _, labels_adv = torch.topk(new_output,2)
        for l in labels_adv:
            if l != self.label[0]:
                break

        return torch.tensor([[ori_output[self.label[0]], ori_output[l]],
         [new_output[self.label[0]], new_output[l]]]).numpy()


    def falsification(self):
        t0 = time.time()
        self.flag = True
        self.attack_pos_all = []
        self.adv_image = None
        self.adv_label = None
        pool = torch.multiprocessing.Pool(self.num_core)  # multiprocessing

        for pos in self.attack_pos:
            net = network(self.model, self.model_name, self.image, self.label, pos,
                               self.layer_gradients, top_dims=self.top_dims, is_cuda=self.is_cuda)
            net.pool = pool
            self.attack_range_3channel(pos)
            all_input_fls = inconfig.partition_input(self.attack_range, pnum=5,
                                                     sp=pos)

            outputSets = []
            outputSets.extend(net.pool.imap(net.regular_reach, all_input_fls))

            all_output_fls = [item for sublist in outputSets for item in sublist]

            self.elapsed_time = time.time() - t0
            print('Number of outputs: ', len(all_output_fls))
            print('Time: ', self.elapsed_time)

            self.attack_pos_all.append(pos)
            self.find_optimal_image(all_output_fls)

            if not self.flag:
                break

        pool.close()


    def find_optimal_image_one_step(self, all_output_fls):
        min_value = 100000
        ap = self.attack_pos_all[-1]
        for afl in all_output_fls:
            vs = afl[0]
            diff = vs[:, self.label].T - vs.T
            diff = np.delete(diff, self.label, 0)

            index = np.argmin(diff)
            min_val = diff.flatten()[index]
            if min_val < min_value:
                min_value = min_val
                min_vs = vs
                v_id = index % diff.shape[1]
                self.image[:, :, ap[0][0]: ap[1][0] + 1, ap[0][1]: ap[1][1] + 1] = torch.tensor(afl[1][v_id, :, :, :])

        new_output = torch.tensor(min_vs[0])
        return new_output


    def find_optimal_image(self, all_output_fls):
        min_value = 100000
        ap = self.attack_pos_all[-1]
        for afl in all_output_fls:
            vs = afl[0]
            diff = vs[:, self.label].T - vs.T
            diff = np.delete(diff, self.label, 0)

            index = np.argmin(diff)
            min_val = diff.flatten()[index]
            if min_val < min_value:
                min_value = min_val
                min_vs = vs
                v_id = index % diff.shape[1]
                self.image[:, :, ap[0][0]: ap[1][0] + 1, ap[0][1]: ap[1][1] + 1] = torch.tensor(afl[1][v_id, :, :, :])

        if min_value<0:

            y = torch.tensor(min_vs[[0]])
            _, self.adv_label = torch.max(y, 1)

            orig_image = torch.squeeze(self.image_orig)
            orig_image = orig_image.permute([1, 2, 0])
            adv_image = torch.squeeze(self.image)
            for k in range(3):
                adv_image[k,:,:] = adv_image[k,:,:]*self.std[k] + self.mean[k]
            self.adv_image = adv_image.permute([1, 2, 0])

            with open(self.savepath+'.pkl', 'wb') as f:
                pickle.dump([self.elapsed_time, self.attack_pos_all, orig_image.numpy(),
                             self.adv_image.numpy(), self.label.numpy(), self.adv_label.numpy()], f)

            plt.imshow(self.adv_image)
            if self.model_name =='cifar10':
                plt.title('Class '+self.classes[self.label[0]]+'; '+'Adv Class '+ self.classes[self.adv_label]+
                           '; '+'Pixels changed '+str(len(self.attack_pos_all)), fontsize=18)
            else:
                plt.title('Class ' + str(self.label.numpy()) + '; ' + 'Adv Class '+str(self.adv_label[0].numpy())+
                           '; '+'Pixels changed '+str(len(self.attack_pos_all)), fontsize=18)
            plt.savefig(self.savepath+'.png')

            print('Adversarial examplea are found!')
            print('Pixels changed: ', len(self.attack_pos_all))
            self.flag = False



    def layer_gradients(self):
        # compute attack position in each layer
        hook_back = []  # create hooks
        for layer in list(self.model._modules.items()):
            if layer[1]._modules != {}:
                for sub_layer in list(layer[1]._modules.items()):
                    if (type(sub_layer[1]).__name__ != 'Dropout2d') and \
                            (type(sub_layer[1]).__name__ != 'Dropout'):
                        hook_back.append(Hook(sub_layer[1], backward=True))
            else:
                hook_back.append(Hook(layer[1], backward=True))

            hook_back.append(Hook())  # for the flatten layer

        hook_back.pop(-1)
        x = Variable(self.image, requires_grad=True)
        y = self.model.forward(x)
        if self.label != y.argmax(axis=-1):
            print('This shouldn\' happen!')
            sys.exit()

        if self.attack_target == None:
            label_temp = self.label
        else:
            label_temp = self.attack_target

        y[0, label_temp].backward(retain_graph=True)

        layer_gradients = []
        for layer in hook_back:
            if layer.input != None:
                for ele in layer.input:
                    if ele != None:
                        layer_gradients.append(torch.squeeze(ele))
                        break
            else:
                layer_gradients.append([])  # for the flatten layer

        image_grad = torch.squeeze(hook_back[0].input[0])

        im_h = self.image.shape[2]
        im_w = self.image.shape[3]
        image_abs_grads = []
        for i in range(im_h - self.pixel_block[0]):
            for j in range(im_w-self.pixel_block[1]):
                tb_grad = image_grad[:,i:(i+self.pixel_block[0]),j:(j+self.pixel_block[1])]
                abs_grads = torch.sum(torch.abs(tb_grad))
                image_abs_grads.append(abs_grads)

        image_abs_grads = torch.tensor(image_abs_grads)
        idx_sorted = image_abs_grads.argsort(descending=True)
        attack_image_pos = []
        for idx in idx_sorted:
            hb = idx//(im_h - self.pixel_block[0])
            wb = idx%(im_w-self.pixel_block[1])
            attack_image_pos.append([np.array([hb, wb]),
                                     np.array([hb+self.pixel_block[0]-1, wb+self.pixel_block[1]-1])])

        return attack_image_pos, layer_gradients
