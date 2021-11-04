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
num_cores = 0 #mp.cpu_count()



class REPAIR:

    def __init__(self, torch_model, properties, data=None):
        self.properties = properties
        self.torch_model = torch_model

        if data is not None:
            self.data = data
        else:
            self.data = self.generate_data()
            # torch.save(self.data, 'acasxu_data.pt')



    def compute_unsafety(self, output_len=100):
        self.ffnn = FFNN(self.torch_model, repair=True)
        all_unsafe_data = []
        num_processors = mp.cpu_count()
        for n, prop in enumerate(self.properties):
            vfl_input = cp.deepcopy(prop.input_set)
            self.ffnn.unsafe_domains = prop.unsafe_domains
            processes = []
            unsafe_data = []
            shared_state = SharedState([vfl_input], num_processors)
            one_worker = Worker(self.ffnn, output_len=output_len)
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


    def generate_data(self, num=100000):  # for training and test
        lbs = self.properties[0].input_ranges[0]
        ubs = self.properties[0].input_ranges[1]

        train_x = torch.tensor([np.random.uniform(lbs[i], ubs[i], num).tolist() for i in range(len(lbs))]).T
        with torch.no_grad():
            train_y = self.torch_model(train_x)
        train_data = self.purify_data([train_x, train_y])

        valid_x = torch.tensor([np.random.uniform(lbs[i], ubs[i], int(num * 0.2)).tolist() for i in range(len(lbs))]).T
        with torch.no_grad():
            valid_y = self.torch_model(valid_x)
        valid_data = self.purify_data([valid_x, valid_y])

        test_x = torch.tensor([np.random.uniform(lbs[i], ubs[i], int(num * 0.2)).tolist() for i in range(len(lbs))]).T
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
        corrected_Ys = []
        original_Xs = []
        for n, subdata in enumerate(unsafe_data):
            if len(subdata) == 0:
                continue

            p = self.properties[n]
            for i in range(len(subdata)):
                orig_x = torch.tensor(subdata[i][0], dtype=torch.float32)
                unsafe_y = torch.tensor(subdata[i][1], dtype=torch.float32)
                violation = 0
                for ufd in p.unsafe_domains:
                    M, vec = ufd[0], ufd[1]
                    target_dims = torch.any(M==0, dim=0)
                    res = torch.mm(M, unsafe_y.T) + vec
                    if torch.any(res > 0,axis=0):
                        continue
                    else:
                        violation += 1
                        assert violation == 1 # only one unsafe domain out of multiple should be violated

                    min_indx = torch.argmax(res)
                    delta_y = M[min_indx] * (-res[min_indx, 0] + epsilon) / (torch.matmul(M[[min_indx]], M[[min_indx]].T))
                    # delta_y = M[min_indx] * (-res[min_indx, 0]*epsilon) / (
                    #     torch.matmul(M[[min_indx]], M[[min_indx]].T))
                    delta_y[target_dims] = 0.0
                    assert not torch.all(delta_y==0.0)
                    safe_y = unsafe_y + delta_y
                    corrected_Ys.append(safe_y)
                    original_Xs.append(orig_x)

        corrected_Ys = torch.cat(corrected_Ys, dim=0)
        original_Xs = torch.cat(original_Xs, dim=0)

        return original_Xs, corrected_Ys



    def compute_accuracy(self, model):
        model.eval()
        predicts = model(self.data.test_data[0])
        pred_actions = torch.argmax(predicts, dim=1)
        actl_actions = torch.argmax(self.data.test_data[1], dim=1)
        accuracy = len(torch.nonzero(pred_actions == actl_actions)) / len(predicts)
        print('Accuracy on the test data: {:.2f}% '.format(accuracy * 100))
        return accuracy



    def repair_model(self, optimizer, loss_fun, savepath, iters=50, batch_size=100, epochs=200):

        all_test_accuracy = []
        accuracy_old = 1.0
        candidate_old = cp.deepcopy(self.torch_model)
        reset_flag = False

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

                unsafe_data = self.compute_unsafety(output_len=1000)
                if np.all([len(sub)==0 for sub in unsafe_data]) and (accuracy_new >= 0.94):
                    print('\nThe accurate and safe candidate model is found !\n')
                    print('\n\n')
                    torch.save(self.torch_model, savepath + "/acasxu_epoch" + str(num) + "_safe.pt")
                    sio.savemat(savepath + '/all_test_accuracy.mat',
                                {'all_test_accuracy': all_test_accuracy})
                    break

                if not np.all([len(sub)==0 for sub in unsafe_data]):
                    # original_Xs, corrected_Ys = self.correct_inputs(unsafe_data, epsilon=5.0)
                    original_Xs, corrected_Ys = self.correct_inputs(unsafe_data)
                    print('Unsafe_inputs: ', len(original_Xs))
                    # train_x = torch.cat((self.data.train_data[0], original_Xs), dim=0)
                    # train_y = torch.cat((self.data.train_data[1], corrected_Ys), dim=0)
                    train_x = original_Xs
                    train_y = corrected_Ys
                else:
                    train_x = self.data.train_data[0]
                    train_y = self.data.train_data[1]

                training_dataset = TensorDataset(train_x.cpu(), train_y.cpu())
                train_loader = DataLoader(training_dataset, batch_size, shuffle=True, num_workers=num_cores)

            reset_flag = False
            print('Start adv training...')
            self.torch_model.train()
            old_weights = cp.deepcopy(self.torch_model[0].weight.data)
            for e in range(epochs):
                print('\rEpoch :'+str(e), end='')
                # print(e, end='\r')
                for batch_idx, (data, target) in enumerate(train_loader):
                    optimizer.zero_grad()
                    predicts = self.torch_model(data)
                    loss = loss_fun(target, predicts)
                    loss.backward()
                    optimizer.step()

            new_weights = cp.deepcopy(self.torch_model[0].weight.data)
            diff = new_weights - old_weights
            xx = torch.any(diff!=0)
            print('\nThe adv training is done\n')
            if num % 1 == 0:
                # torch.save(candidate.state_dict(), savepath + "/acasxu_epoch" + str(num) + ".pt")
                torch.save(self.torch_model, savepath + "/acasxu_epoch" + str(num) + ".pt")


class DATA:
    def __init__(self, train_data, valid_data, test_data):
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data

