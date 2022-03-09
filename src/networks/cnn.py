import torch
import copy
import time
import torch.nn as nn
import cubelattice3c as cl3c
import numpy as np
import itertools
import copy as cp


class Flatten(nn.Module):
    def forward(self, input):
        return torch.flatten(input,1)

class network:
    def __init__(self, net, name, image, label, attack_pos, layer_gradients,
                 is_cuda=False, top_dims=1):

        self.pool = None
        self.class_label = None
        self.flatten_pos = None # the position of the flatten function in layers
        self.is_cuda = is_cuda
        self.net_name = name
        self.layer_gradients = layer_gradients
        self.top_dims = top_dims # top_dims<1 indicates under approximation
        self.label = label
        self.start_time = 0

        for param in net.parameters():
            param.requires_grad = False

        if name == 'vgg16':
            self.sequential = self.forward_layer_sequential_vgg16(net)
        elif name == 'cifar10':
            self.sequential = self.forward_layer_sequential_cifar10(net)

        self.layer_num = len(self.sequential) # the number of layers

        if self.net_name == 'vgg16':
            self.all_inputs = self.forward_layer_output_vgg16(image)

        elif self.net_name == 'cifar10':
            self.all_inputs = self.forward_layer_output_cifar10(image)


        self.all_range_pos, self.all_attack_pos = self.layers_range(attack_pos)
        if self.top_dims < 1:
            self.indices_selected_layers = self.get_target_indices()

        self._layer = None
        self._image_frame = None
        self._slice_blocks = None


    def forward_layer_sequential_cifar10(self, net):
        net_layers = []
        if hasattr(net, 'conv_layer'):
            for i in range(len(net.conv_layer)):
                type_name = type(net.conv_layer[i]).__name__
                if type_name == 'Dropout2d':
                    continue
                else:
                    net_layers.append(net.conv_layer[i])

            net_layers.append(Flatten())
            self.flatten_pos = len(net_layers)-1
            for i in range(len(net.fc_layer)):
                type_name = type(net.fc_layer[i]).__name__
                if type_name == 'Dropout':
                    continue
                else:
                    net_layers.append(net.fc_layer[i])
        else:
            for i in range(len(net.sequential)):
                type_name = type(net.sequential[i]).__name__
                if type_name == 'Dropout2d' or type_name == 'Dropout':
                    continue
                if type_name == 'Flatten':
                    self.flatten_pos = i

                net_layers.append(net.sequential[i])

        if self.is_cuda:
            sequential_layers = nn.Sequential(*net_layers).eval().cuda()
        else:
            sequential_layers = nn.Sequential(*net_layers).eval()

        for param in sequential_layers.parameters():
            param.requires_grad = False

        return sequential_layers


    def forward_layer_sequential_vgg16(self, net):
        net_layers = []
        for i in range(len(net.features)):
            net_layers.append( net.features[i])

        net_layers.append(Flatten())
        self.flatten_pos = len(net_layers)-1
        net_layers.append(net.classifier[0])
        net_layers.append(net.classifier[1])
        net_layers.append(net.classifier[3])

        net_layers.append(net.classifier[4])
        net_layers.append(net.classifier[6])
        if self.is_cuda:
            sequential_layers = nn.Sequential(*net_layers).eval().cuda()
        else:
            sequential_layers = nn.Sequential(*net_layers).eval()

        for param in sequential_layers.parameters():
            param.requires_grad = False

        return sequential_layers


    def forward_layer_output_cifar10(self, image):
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


    def forward_layer_output_vgg16(self, image):
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

        self.class_label = np.argmax(all_inputs[-1], 1)
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
            i = 0
            while i<all_fls_len:
                im_fl = all_fls[0]
                all_fls.pop(0)
                try: #avoid numerical error
                    im_fl_output = self.relu_layer1(im_fl, neurons, False)
                    all_fls.extend(im_fl_output)
                    i += 1
                except:
                    i+=1
                    continue


        elif type_name == 'MaxPool2d':

            r2p = self.all_range_pos[self._layer]
            attack_pos = self.all_attack_pos[self._layer]
            pos = attack_pos - r2p[0]
            self._slice_blocks = self.get_slice_blocks(r2p)
            self._image_frame = self.all_inputs[self._layer][:, :, r2p[0][0]:r2p[1][0] + 1, r2p[0][1]:r2p[1][1] + 1]

            all_fls_length = len(all_fls)
            i = 0
            while i<all_fls_length:
                im_fl = all_fls[0]
                all_fls.pop(0)
                data_frame = self._image_frame.repeat(im_fl.vertices.shape[0], 0)
                data_frame[:, :, pos[0][0]:pos[1][0] + 1, pos[0][1]:pos[1][1] + 1] = cp.deepcopy(im_fl.vertices)
                im_fl.vertices = data_frame
                blocks = np.array([])
                try: #avoid numerical errors
                    im_fl_output = self.maxpool2d_layer(im_fl, blocks, False)
                    all_fls.extend(im_fl_output)
                    i+=1
                except:
                    i+=1
                    continue

        elif type_name == 'Flatten':
            self._image_frame = self.all_inputs[self._layer]

            for im_fl in all_fls:
                self.flatten_layer(im_fl)

        elif type_name == 'ReLU' and self._layer > self.flatten_pos:
            neurons = np.array([])
            all_fls_length = len(all_fls)
            i = 0
            while i < all_fls_length:
                im_fl = all_fls[0]
                all_fls.pop(0)
                try:
                    im_fl_output = self.relu_layer2(im_fl, neurons, False)
                    all_fls.extend(im_fl_output)
                    i+=1
                except:
                    i+=1
                    continue

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

        # return all_fls

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
        clattice = cl3c.cubelattice3c()
        image_fl = clattice.to_facelattice()
        # if image.is_cuda:
        #     image_fl.to_cuda()

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

