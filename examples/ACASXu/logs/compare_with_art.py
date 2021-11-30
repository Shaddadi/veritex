import sys

import numpy as np
sys.path.insert(0, '../../../src')
sys.path.insert(0, '../')
from load_onnx import load_ffnn_onnx
from acasxu_repair_list import *
import scipy.io as sio
from nnet_file import NNet
import copy as cp
from ffnn import FFNN
import multiprocessing as mp
from worker import Worker
from shared import SharedState
import torch
import glob
import os

repaired_nets = [[1,9],[2,1],[2,2],[2,3],[2,4],[2,5],[2,6],[2,7],[2,8],[2,9],
                 [3,1],[3,2],[3,4],[3,5],[3,6],[3,7],[3,8],[3,9],[4,1],[4,3],
                 [4,4],[4,5],[4,6],[4,7],[4,8],[4,9],[5,1],[5,2],[5,3],[5,4],
                 [5,5],[5,6],[5,7],[5,8],[5,9]]



def extract_weights(torch_model):
    weights = []
    bias = []
    for layer in torch_model:
        if isinstance(layer, torch.nn.Linear):
            weights.append(layer.weight.data.numpy())
            bias.append(layer.bias.data.numpy())

    return weights, bias



def get_log_info(log_path):
    def get_time(l):
        t = ""
        start = False
        for ch in l:
            if start:
                t += ch
            if ch == " " and start:
                break
            elif ch == "(":
                start = True

        return float(t)

    with open(log_path) as file:
        lines = file.readlines()
        lines = [line.rstrip() for line in lines]
        net_IDs = []
        accurcys = []
        run_times = []
        for indx, l in enumerate(lines):
            if l[32:45] == 'For AcasNetID':
                net_IDs.append([int(l[46]), int(l[48])])
            elif l[32:55] == 'Accuracy at every epoch':
                accus = np.fromstring(l[58:-1],sep=',')
                accurcys.append(accus)
                t = get_time(lines[indx+1])
                run_times.append(t)

        assert len(net_IDs)==len(accurcys) and len(net_IDs)==len(run_times)
        return net_IDs, accurcys, run_times



def collect_accuracy_time():
    art_log_path_refine = ["art_test_goal_safety/test_goal_safety-2021-11-27-21-56-33.log",
                    "art_test_goal_safety/test_goal_safety-2021-11-27-23-59-41.log"]
    art_log_path_no_refine = ["art_test_goal_safety_no_refine/test_goal_safety-2021-11-27-23-03-15.log",
                              "art_test_goal_safety_no_refine/test_goal_safety-2021-11-27-23-57-42.log"]

    results_refine = []
    for log_path in art_log_path_refine:
        net_IDs, accurcys, run_times = get_log_info(log_path)
        for indx, net in enumerate(net_IDs):
            net_name = 'net'+str(net[0])+str(net[1])
            results_refine.append([accurcys[indx], run_times[indx]])

    results_no_refine = []
    for log_path in art_log_path_no_refine:
        net_IDs, accurcys, run_times = get_log_info(log_path)
        for indx, net in enumerate(net_IDs):
            net_name = 'net'+str(net[0])+str(net[1])
            results_no_refine.append([accurcys[indx], run_times[indx]])

    return results_refine, results_no_refine




def compute_linear_regions(model, properties):
    if isinstance(model, torch.nn.Sequential):
        ffnn = FFNN(model, linear_regions=True)
    else:
        biases = [np.array([bia]).T for bia in model.biases]
        ffnn = FFNN([model.weights, biases], linear_regions=True)

    num_exact_output = 0
    num_processors = mp.cpu_count()
    for n, prop in enumerate(properties):
        vfl_input = cp.deepcopy(prop[0].input_set)
        ffnn.unsafe_domains = prop[0].unsafe_domains
        processes = []
        exact_output = []
        shared_state = SharedState([vfl_input], num_processors)
        one_worker = Worker(ffnn)
        for index in range(num_processors):
            p = mp.Process(target=one_worker.main_func, args=(index, shared_state))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()

        num_exact_output += shared_state.outputs_len.value

    return num_exact_output


def compute_expressibility():
    for indx, item in enumerate(repair_list):
        if indx == 0:
            continue
        i, j = item[0][0], item[0][1]
        print('Neural Network',i,j)
        properties_repair = item[1]

        nn_original_path = "../nets/ACASXU_run2a_" + str(i) + "_" + str(j) + "_batch_2000.onnx"
        original_model = load_ffnn_onnx(nn_original_path)
        ori_lregion = compute_linear_regions(original_model, properties_repair)

        nn_our_path = "logs_lr0.001_epochs200/nnet" + str(i) + str(j) + "_lr0.001_epochs200"
        list_of_files_path = glob.glob(nn_our_path + '/*')
        safe_net_path = max(list_of_files_path, key=os.path.getctime)
        assert safe_net_path[-7:-3] == 'safe'
        our_safe_model = torch.load(safe_net_path)
        our_lregion = compute_linear_regions(our_safe_model, properties_repair)

        nn_art_path_refine = "art_test_goal_safety/repaired_network_"+str(i)+str(j)+"_safe.nnet"
        art_safe_model_refine = NNet(nn_art_path_refine)
        art_lregion_refine = compute_linear_regions(art_safe_model_refine, properties_repair)


        try:
            nn_art_path_refine = "art_test_goal_safety_no_refine/repaired_network_" + str(i) + str(j) + "_safe.nnet"
            art_safe_model_no_refine = NNet(nn_art_path_refine)
        except:
            nn_art_path_refine = "art_test_goal_safety_no_refine/repaired_network_" + str(i) + str(j) + "_unsafe.nnet"
            art_safe_model_no_refine = NNet(nn_art_path_refine)

        art_lregion_no_refine = compute_linear_regions(art_safe_model_no_refine, properties_repair)







def compute_weights_ratio():
    for anet in repaired_nets:
        i, j = anet[0],anet[1]
        nn_our_path = "logs_lr0.001_epochs200/nnet" + str(i) + str(j) + "_lr0.001_epochs200"
        list_of_files_path = glob.glob(nn_our_path+'/*')
        safe_net_path = max(list_of_files_path, key=os.path.getctime)
        assert safe_net_path[-7:-3] == 'safe'
        our_safe_model = torch.load(safe_net_path)
        our_weights, our_bias = extract_weights(our_safe_model)


        nn_original_path = "../nets/ACASXU_run2a_" + str(i) + "_" + str(j) + "_batch_2000.onnx"
        original_model = load_ffnn_onnx(nn_original_path)
        original_weights, original_bias = extract_weights(original_model)

        nn_art_path_refine = "art_test_goal_safety/repaired_network_"+str(i)+str(j)+"_safe.nnet"
        art_safe_model_refine = NNet(nn_art_path_refine)

        try:
            nn_art_path_refine = "art_test_goal_safety_no_refine/repaired_network_" + str(i) + str(j) + "_safe.nnet"
            art_safe_model_no_refine = NNet(nn_art_path_refine)
        except:
            nn_art_path_refine = "art_test_goal_safety_no_refine/repaired_network_" + str(i) + str(j) + "_unsafe.nnet"
            art_safe_model_no_refine = NNet(nn_art_path_refine)

        square_sum_ours = 0
        square_sum_art_refine = 0
        square_sum_art_no_refine = 0
        square_sum_base = 0
        for layer in range(len(original_weights)):
            square_sum_base += np.sum(np.square(original_weights[layer])) + np.sum(np.square(original_bias[layer]))

            our_diff = np.sum(np.square(our_weights[layer] - original_weights[layer])) \
                       + np.sum(np.square(our_bias[layer]-original_bias[layer]))
            square_sum_ours += our_diff

            art_refine_diff = np.sum(np.square(art_safe_model_refine.weights[layer] - original_weights[layer])) \
                       + np.sum(np.square(art_safe_model_refine.biases[layer] - original_bias[layer]))
            square_sum_art_refine += art_refine_diff

            art_no_refine_diff = np.sum(np.square(art_safe_model_no_refine.weights[layer] - original_weights[layer])) \
                       + np.sum(np.square(art_safe_model_no_refine.biases[layer] - original_bias[layer]))
            square_sum_art_no_refine += art_no_refine_diff


        our_deviation = np.sqrt(square_sum_ours)/np.sqrt(square_sum_base)
        art_refine = np.sqrt(square_sum_art_refine)/np.sqrt(square_sum_base)
        art_no_refine = np.sqrt(square_sum_art_no_refine)/np.sqrt(square_sum_base)

        return our_deviation, art_refine, art_no_refine


if __name__ == "__main__":
    # results_refine, results_no_refine = collect_accuracy_time()
    our_deviation, art_refine, art_no_refine = compute_weights_ratio()
    # sio.savemat('art_accuracy_time.mat', {'results_refine': results_refine,
    #                                       'results_no_refine': results_no_refine})
    # sio.savemat('weights_deviation.mat', {'our_deviation': our_deviation,
    #                                       'art_refine_dev': art_refine,
    #                                       'art_no_refine_dev': art_no_refine})

    # compute_expressibility()






