from ffnn import FFNN
import torch
import numpy as np
import copy as cp
import multiprocessing as mp
from torch.utils.data import TensorDataset, DataLoader
from worker import Worker
from shared import SharedState
import scipy.io as sio
num_cores = mp.cpu_count()



class REPAIR:

    def __init__(self, torch_model, properties, data=None):
        self.properties = properties
        if data is not None:
            self.data = data
        else:
            self.data = self.generate_data()

        self.torch_model = torch_model
        self.ffnn = FFNN(torch_model, repair=True)



    def compute_unsafety(self):
        all_unsafe_data = []
        num_processors = mp.cpu_count()
        for n, prop in enumerate(self.properties):
            vfl_input = cp.deepcopy(prop.input_set)
            self.ffnn.unsafe_domains = prop.unsafe_domains
            processes = []
            unsafe_data = []
            shared_state = SharedState([vfl_input], num_processors)
            one_worker = Worker(self.ffnn, output_len=100)
            for index in range(num_processors):
                p = mp.Process(target=one_worker.main_func, args=(index, shared_state))
                processes.append(p)
                p.start()

            for p in processes:
                p.join()

            while not shared_state.outputs.empty():
                unsafe_data.append(shared_state.outputs.get())

            all_unsafe_data.append(all_unsafe_data)

        return all_unsafe_data


    def generate_data(self, num=100000):  # for training and test
        lbs = self.properties.input_ranges[0]
        ubs = self.properties.input_ranges[1]

        train_x = torch.tensor([np.random.uniform(lbs[i], ubs[i], num).tolist() for i in range(len(lbs))]).T
        train_y = self.torch_model(train_x)
        train_data = self.purify_data([train_x, train_y])

        valid_x = torch.tensor([np.random.uniform(lbs[i], ubs[i], int(num * 0.2)).tolist() for i in range(len(lbs))]).T
        valid_y = self.torch_model(valid_x)
        valid_data = self.purify_data([valid_x, valid_y])

        test_x = torch.tensor([np.random.uniform(lbs[i], ubs[i], int(num * 0.2)).tolist() for i in range(len(lbs))]).T
        test_y = self.torch_model(test_x)
        test_data = self.purify_data([test_x, test_y])

        return DATA(train_data, valid_data, test_data)


    def purify_data(self, data): # check and remove unsafe data
        data_x = data[0]
        data_y = data[1]
        for p in self.properties:
            lb, ub = p[0][0], p[0][1]
            M, vec = p[1][0], p[1][1]
            bools = torch.ones(len(data_x), dtype=torch.bool)
            if torch.cuda.is_available():
                bools = bools.cuda()

            for n in range(len(lb)):
                lbx, ubx = lb[n], ub[n]
                x = data_x[:, n]
                bools = (x > lbx) & (x < ubx) & bools

            if len(x) == 0:
                break

            outs = torch.dot(M, x.T) + vec
            out_bools = torch.all(outs, dim=0)
            safe_indx = torch.nonzero(~out_bools)
            if torch.cuda.is_available():
                safe_indx = safe_indx.cuda()

            data_x = data_x[safe_indx]
            data_y = data_y[safe_indx]

        return [data_x, data_y]



    def correct_inputs(self, unsafe_data, epsilon=0.01):
        corrected_ys = []
        original_Xs = []
        for n, aset in enumerate(unsafe_data):
            if len(aset[0]) == 0:
                continue

            X_unsafe = aset[0]
            Y_unsafe = aset[1]
            p = self.properties[n]
            unsafe_domains = p.unsafe_domains
            M, vec = torch.tensor(p[1][0]), torch.tensor(p[1][1])
            if torch.cuda.is_available():
                M, vec = M.cuda(), vec.cuda()

            for i in range(len(Y_unsafe)):
                unsafe_y = Y_unsafe[[i]]
                res = torch.matmul(M, unsafe_y.T) + vec
                if torch.any(res > 0):
                    continue

                min_indx = torch.argmax(res)

                delta_y = M[min_indx] * (-res[min_indx, 0] + epsilon) / (torch.matmul(M[[min_indx]], M[[min_indx]].T))
                safe_y = unsafe_y + delta_y
                corrected_ys.append(safe_y)
                original_Xs.append(X_unsafe[[i]])

        corrected_ys = torch.cat(corrected_ys, dim=0)
        original_Xs = torch.cat(original_Xs, dim=0)

        return original_Xs, corrected_ys



    def compute_accuracy(self, model):
        model.eval()
        predicts = model(self.data.test_x)
        pred_actions = torch.argmax(predicts, dim=1)
        actl_actions = torch.argmax(self.data.test_y, dim=1)
        accuracy = len(torch.nonzero(pred_actions == actl_actions)) / len(predicts)
        print('  Accuracy on the test data: {:.2f}% '.format(accuracy * 100))
        return accuracy



    def repair_model(self, candidate, optimizer, loss_fun, savepath, iters=200, batch_size=1000):

        all_test_accuracy = []
        all_reach_vfls = []
        all_unsafe_vfls = []
        for num in range(iters):
            print('Iteration of repairing: ', num)
            unsafe_data = self.compute_unsafety()
            accuracy = self.compute_accuracy(candidate.eval())
            if np.all([len(aset[0]) == 0 for aset in unsafe_data]) and (accuracy >= 0.94):
                print('\nThe accurate and safe candidate model is found !\n')
                print('\n\n')
                torch.save(candidate.state_dict(), savepath + "/acasxu_epoch" + str(num) + "_safe.pt")
                sio.savemat(savepath + '/all_test_accuracy.mat',
                            {'all_test_accuracy': all_test_accuracy})
                sio.savemat(savepath + '/reach_sets.mat',
                            {'all_reach_vfls': all_reach_vfls, 'all_unsafe_vfls': all_unsafe_vfls})
                break

            if not np.all([len(aset[0])==0 for aset in unsafe_data]):
                unsafe_xs, corrected_ys = self.correct_inputs(unsafe_data, epsilon=0.01)
                print('  Unsafe_inputs: ', len(unsafe_xs))
                train_x = torch.cat((self.data.train_data[0], unsafe_xs), dim=0)
                train_y = torch.cat((self.data.train_data[1], corrected_ys), dim=0)
            else:
                train_x = self.data.train_data[0]
                train_y = self.data.train_data[1]

            training_dataset = TensorDataset(train_x.cpu(), train_y.cpu())
            train_loader = DataLoader(training_dataset, batch_size, shuffle=True, num_workers=num_cores)

            print('  Start training...')
            candidate.train()
            for batch_idx, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                predicts = candidate(data.cuda())
                loss = loss_fun(target.cuda(), predicts)
                loss.backward()
            print('  The training is done\n')

            if num % 1 == 0:
                torch.save(candidate.state_dict(), savepath + "/acasxu_epoch" + str(num) + ".pt")


class DATA:
    def __init__(self, train_data, valid_data, test_data):
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data

