import time

from ffnn import FFNN
import torch
import numpy as np
import copy as cp
import multiprocessing as mp
from torch.utils.data import TensorDataset, DataLoader
from worker import Worker
from shared import SharedState
import scipy.io as sio
import torch.optim as optim
import torch.nn as nn
import pickle
import sys
num_cores = 0 #mp.cpu_count()



class REPAIR:

    def __init__(self, torch_model, properties_repair, data=None, output_limit=1000):
        self.properties = [item[0] for item in properties_repair]
        self.corrections = [item[1] for item in properties_repair]
        self.torch_model = torch_model
        self.output_limit = output_limit

        if data is not None:
            self.data = data
        else:
            self.data = self.generate_data()
            # torch.save(self.data, 'acasxu_data.pt')

    def fast_verify_layers(self):
        num_processors = mp.cpu_count()
        for n, prop in enumerate(self.properties):
            vfl_input = cp.deepcopy(prop.input_set)
            affine_relu_lens = int((len(self.torch_model) - 1) / 2)
            for layer in range(affine_relu_lens):
                layer_model = self.torch_model[2*layer:(2*layer+2)]
                self.ffnn = FFNN(layer_model, exact_output=True)
                self.ffnn.unsafe_domains = prop.unsafe_domains
                processes = []
                output_sets = []
                shared_state = SharedState([vfl_input], num_processors)
                one_worker = Worker(self.ffnn, output_len=self.output_limit)
                for index in range(num_processors):
                    p = mp.Process(target=one_worker.main_func, args=(index, shared_state))
                    processes.append(p)
                    p.start()

                for p in processes:
                    p.join()

                while not shared_state.outputs.empty():
                    output_sets.append(shared_state.outputs.get())

                # for item in output_sets:



    def compute_unsafety(self):
        self.ffnn = FFNN(self.torch_model, repair=True)
        all_unsafe_data = []
        num_processors = mp.cpu_count()
        for n, prop in enumerate(self.properties):
            vfl_input = cp.deepcopy(prop.input_set)
            # from utils import split_bounds
            # from boxdomain import BoxDomain
            # lbs_input, ubs_input = prop.lbs, prop.ubs
            # sub_vzono_sets = split_bounds(lbs_input, ubs_input, num=2)
            # boxes = [BoxDomain(item[1][0], item[1][1]) for item in sub_vzono_sets]
            # new_sub_vzono_sets = [item.toFacetVertex() for item in boxes]
            self.ffnn.unsafe_domains = prop.unsafe_domains
            processes = []
            unsafe_data = []
            shared_state = SharedState([vfl_input], num_processors)
            one_worker = Worker(self.ffnn, output_len=self.output_limit)
            for index in range(num_processors):
                p = mp.Process(target=one_worker.main_func, args=(index, shared_state))
                processes.append(p)
                p.start()

            for p in processes:
                p.join()

            while not shared_state.outputs.empty():
                unsafe_data.append(shared_state.outputs.get())

            all_unsafe_data.append(unsafe_data)

        return all_unsafe_data


    def generate_data(self, num=10000):  # for training and test
        lbs = self.properties[0].input_ranges[0]
        ubs = self.properties[0].input_ranges[1]

        train_x = torch.tensor([np.random.uniform(lbs[i], ubs[i], num).tolist() for i in range(len(lbs))]).T
        with torch.no_grad():
            train_y = self.torch_model(train_x)
        train_data = self.purify_data([train_x, train_y])

        valid_x = torch.tensor([np.random.uniform(lbs[i], ubs[i], int(num * 0.5)).tolist() for i in range(len(lbs))]).T
        with torch.no_grad():
            valid_y = self.torch_model(valid_x)
        valid_data = self.purify_data([valid_x, valid_y])

        test_x = torch.tensor([np.random.uniform(lbs[i], ubs[i], int(num * 0.5)).tolist() for i in range(len(lbs))]).T
        with torch.no_grad():
            test_y = self.torch_model(test_x)
        test_data = self.purify_data([test_x, test_y])

        return DATA(train_data, valid_data, test_data)


    def purify_data(self, data): # check and remove unsafe data
        data_x = data[0]
        data_y = data[1]
        for p in self.properties:
            lb, ub = p.lbs, p.ubs
            for ufd in p.unsafe_domains:
                M, vec = ufd[0], ufd[1]
                bools = torch.ones(len(data_x), dtype=torch.bool)
                # if torch.cuda.is_available():
                #     bools = bools.cuda()

                for n in range(len(lb)):
                    lbx, ubx = lb[n], ub[n]
                    x = data_x[:, n]
                    bools = (x > lbx) & (x < ubx) & bools

                if not torch.any(bools):
                    continue

                outs = torch.mm(M, data_y.T) + vec
                out_bools = torch.all(outs<=0, dim=0) & bools
                if not torch.any(out_bools):
                    continue

                safe_indx = torch.nonzero(~out_bools)[:,0]
                # if torch.cuda.is_available():
                #     safe_indx = safe_indx.cuda()
                data_x = data_x[safe_indx]
                data_y = data_y[safe_indx]

        return [data_x, data_y]


    def correct_inputs(self, unsafe_data, epsilon=0.001):
        # for classification
        corrected_Ys = []
        original_Xs = []
        length_unsafe_data = 0
        for n, subdata in enumerate(unsafe_data):
            length_unsafe_data += len(subdata)
            if len(subdata) == 0:
                continue

            p = self.properties[n]
            for i in range(len(subdata)):
                orig_x = torch.tensor(subdata[i][0], dtype=torch.float32)
                unsafe_y = torch.tensor(subdata[i][1], dtype=torch.float32)
                label = torch.argmax(unsafe_y*(-1))
                violation = 0

                target_dim_all = []
                for ufd in p.unsafe_domains:
                    zz = torch.all(ufd[0] != 0, dim=0)
                    xx = torch.nonzero(torch.all(ufd[0] != 0, dim=0))
                    target_dim_temp = torch.nonzero(torch.all(ufd[0] != 0, dim=0))[0][0]
                    target_dim_all.append(target_dim_temp)

                target_dim_all.append(label)

                for ufd_id, ufd in enumerate(p.unsafe_domains):
                    M, vec = ufd[0], ufd[1]
                    target_dim = target_dim_all[ufd_id]
                    res = torch.mm(M, unsafe_y.T) + vec
                    if torch.any(res > 0,dim=0):
                        continue
                    else:
                        violation += 1
                        assert violation == 1 # only one unsafe domain out of multiple should be violated

                    max_indx = torch.argmax(res)
                    if M[max_indx][target_dim] > 0.0: # or unsafe_y[0][target_dim] == torch.max(unsafe_y)
                        delta_y = (-res[max_indx, 0] + epsilon) / M[max_indx][target_dim]
                        unsafe_y[0][target_dim] = unsafe_y[0][target_dim] + delta_y  # safe y
                    else: # unsafe_y[0][target_dim] < torch.max(unsafe_y[0])
                        sorted, indices = torch.sort(res[:,0], descending=True)
                        for min_indx2 in indices:
                            target_dim2 = torch.nonzero(M[min_indx2] == 1)[0][0]
                            if target_dim2 not in target_dim_all:
                                delta_y = (-res[min_indx2, 0] + epsilon) / M[min_indx2][target_dim2]
                                unsafe_y[0][target_dim2] = unsafe_y[0][target_dim2] + delta_y  # safe y
                                break

                    res = torch.mm(M, unsafe_y.T) + vec
                    assert torch.any(res > 0, dim=0)

                    # # make this classification comply the most common classification by switching values
                    # max_indx = torch.argmax(unsafe_y)
                    # if max_indx != self.optimal_dim:
                    #     actual_max = cp.deepcopy(unsafe_y[0][max_indx])
                    #     prefered_max = cp.deepcopy(unsafe_y[0][self.optimal_dim])
                    #     unsafe_y[0][self.optimal_dim] = actual_max
                    #     unsafe_y[0][max_indx] = prefered_max

                    corrected_Ys.append(unsafe_y)
                    original_Xs.append(orig_x)

        corrected_Ys = torch.cat(corrected_Ys, dim=0)
        original_Xs = torch.cat(original_Xs, dim=0)

        print('Unsafe_inputs: ', length_unsafe_data)
        return original_Xs, corrected_Ys


    def correct_unsafe_data(self, unsafe_data):
        length_unsafe_data = 0
        corrected_Ys = []
        original_Xs = []
        for n, subdata in enumerate(unsafe_data):
            length_unsafe_data += len(subdata)
            if len(subdata) == 0:
                continue

            correction = self.corrections[n]
            safe_x, safe_y = correction(subdata)
            original_Xs.append(safe_x)
            corrected_Ys.append(safe_y)

        corrected_Ys = torch.cat(corrected_Ys, dim=0)
        original_Xs = torch.cat(original_Xs, dim=0)
        print('Unsafe_inputs: ', length_unsafe_data)
        return original_Xs, corrected_Ys



    def compute_accuracy(self, model):
        model.eval()
        predicts = model(self.data.test_data[0]) * (-1)  # The minimum is the predication
        pred_actions = torch.argmax(predicts, dim=1)
        actl_actions = torch.argmax(self.data.test_data[1] * (-1), dim=1)
        actions_times = torch.tensor([len(torch.nonzero(actl_actions==n)[:,0]) for n in range(predicts.shape[1])])
        self.optimal_dim = torch.argmax(actions_times)
        accuracy = len(torch.nonzero(pred_actions == actl_actions)) / len(predicts)
        print('Accuracy on the test data: {:.2f}% '.format(accuracy * 100))
        return accuracy



    def repair_model(self, optimizer, loss_fun, savepath, iters=100, batch_size=2000, epochs=200):
        t0 = time.time()
        all_test_accuracy = []
        accuracy_old = 1.0
        candidate_old = cp.deepcopy(self.torch_model)
        reset_flag = False
        t0 = time.time()
        for num in range(iters):
            print('Iteration of repairing: ', num)
            accuracy_new = self.compute_accuracy(self.torch_model)
            if accuracy_old - accuracy_new > 0.1:
                print('A large drop of accuracy\n')
                self.torch_model = cp.deepcopy(candidate_old)
                lr = optimizer.param_groups[0]['lr'] * 0.8
                print('lr :', lr)
                optimizer = optim.SGD(self.torch_model.parameters(), lr=lr, momentum=0.9)
                reset_flag = True
                continue

            if not reset_flag:
                candidate_old = cp.deepcopy(self.torch_model)
                accuracy_old = accuracy_new
                all_test_accuracy.append(accuracy_new)

                unsafe_data = self.compute_unsafety()
                if np.all([len(sub)==0 for sub in unsafe_data]):
                    print('The accurate and safe candidate model is found !')
                    torch.save(self.torch_model, savepath + "/acasxu_epoch" + str(num) + "_safe.pt")
                    print('Running time: ', time.time()-t0)
                    print('\n\n')
                    sio.savemat(savepath + '/all_test_accuracy.mat',
                                {'all_test_accuracy': all_test_accuracy, 'time': time.time()-t0})
                    break

                if not np.all([len(sub)==0 for sub in unsafe_data]):
                    # original_Xs, corrected_Ys = self.correct_inputs(unsafe_data)
                    original_Xs, corrected_Ys = self.correct_unsafe_data(unsafe_data)

                    train_x = original_Xs
                    train_y = corrected_Ys
                else:
                    train_x = self.data.train_data[0]
                    train_y = self.data.train_data[1]

                training_dataset = TensorDataset(train_x.cuda(), train_y.cuda())
                train_loader = DataLoader(training_dataset, batch_size, shuffle=True, num_workers=num_cores)

            reset_flag = False
            print('Start adv training...')
            self.torch_model.train()
            old_weights = cp.deepcopy(self.torch_model[0].weight.data)
            self.torch_model.cuda()
            for e in range(epochs):
                print('\rEpoch :'+str(e), end='')
                # print(e, end='\r')
                # average_loss = 0
                for batch_idx, (data, target) in enumerate(train_loader):
                    optimizer.zero_grad()
                    predicts = self.torch_model(data)
                    loss = loss_fun(target, predicts)
                    # average_loss += loss.data
                    loss.backward()
                    optimizer.step()

                # print('Averaged loss :', average_loss/(batch_idx+1))

            self.torch_model.cpu()
            new_weights = cp.deepcopy(self.torch_model[0].weight.data)
            diff = new_weights - old_weights
            xx = torch.any(diff!=0)
            print('\nThe adv training is done\n')
            print('Running time: ', time.time() - t0)
            if num % 1 == 0:
                # torch.save(candidate.state_dict(), savepath + "/acasxu_epoch" + str(num) + ".pt")
                torch.save(self.torch_model, savepath + "/acasxu_epoch" + str(num) + ".pt")


class DATA:
    def __init__(self, train_data, valid_data, test_data):
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data

