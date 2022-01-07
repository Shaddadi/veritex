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
    art_log_path_refine = ["art_test_goal_safety/test_goal_safety-2021-11-27-23-59-41.log",
                            "art_test_goal_safety/test_goal_safety-2021-11-27-21-56-33.log"]
    art_log_path_no_refine = ["art_test_goal_safety_no_refine/test_goal_safety-2021-11-27-23-57-42.log",
                              "art_test_goal_safety_no_refine/test_goal_safety-2021-11-27-23-03-15.log"]

    results_refine = []
    for log_path in art_log_path_refine:
        net_IDs, accurcys, run_times = get_log_info(log_path)
        for indx, net in enumerate(net_IDs):
            if net in safe_nnet_list:
                continue
            net_name = 'net'+str(net[0])+str(net[1])
            results_refine.append([accurcys[indx], run_times[indx]])

    results_no_refine = []
    for log_path in art_log_path_no_refine:
        net_IDs, accurcys, run_times = get_log_info(log_path)
        for indx, net in enumerate(net_IDs):
            if net in safe_nnet_list:
                continue
            net_name = 'net'+str(net[0])+str(net[1])
            results_no_refine.append([accurcys[indx], run_times[indx]])

    return results_refine, results_no_refine


def compute_linear_region_vs(model, properties):
    if isinstance(model, torch.nn.Sequential):
        ffnn = FFNN(model, exact_output=True)
    else:
        biases = [np.array([bia]).T for bia in model.biases]
        ffnn = FFNN([model.weights, biases], exact_output=True)

    input_vs = []
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

        while not shared_state.outputs.empty():
            output_set = shared_state.outputs.get()
            input_vs.append(output_set.vertices[0])

    return input_vs


def compute_reachable_set_distance():
    if os.path.isfile('output_distance.mat'):
       all_diffs =  sio.loadmat('output_distance.mat')
       all_diff_our_outputs = list(all_diffs['all_diff_our_outputs'][0,:])
       all_diff_art_refine_outputs = list(all_diffs['all_diff_art_refine_outputs'][0,:])
       all_diff_art_no_refine_outputs = list(all_diffs['all_diff_art_no_refine_outputs'][0,:])
       start_id = len(all_diff_our_outputs)
    else:
        all_diff_our_outputs = []
        all_diff_art_refine_outputs = []
        all_diff_art_no_refine_outputs = []
        start_id = 0


    for indx, item in enumerate(repair_list):
        if indx == 0 or indx <= start_id:
            continue

        i, j = item[0][0], item[0][1]
        print('Neural Network',i,j)
        properties_repair = item[1]

        nn_original_path = "../nets/ACASXU_run2a_" + str(i) + "_" + str(j) + "_batch_2000.onnx"
        original_model = load_ffnn_onnx(nn_original_path)
        all_inputs = compute_linear_region_vs(original_model, properties_repair)

        all_inputs = torch.tensor(np.array(all_inputs),dtype=torch.float32)
        with torch.no_grad():
            orig_outputs = original_model(all_inputs)

        nn_our_path = "logs_lr0.001_epochs200/nnet" + str(i) + str(j) + "_lr0.001_epochs200"
        list_of_files_path = glob.glob(nn_our_path + '/*')
        for safe_net_path in list_of_files_path:
            if safe_net_path[-7:-3] == 'safe':
                break
        our_safe_model = torch.load(safe_net_path)
        with torch.no_grad():
            our_outputs = our_safe_model(all_inputs)
        diff_our_outputs = torch.mean((our_outputs-orig_outputs).norm(dim=1, p=2))
        all_diff_our_outputs.append(diff_our_outputs)

        nn_art_path_refine = "art_test_goal_safety/repaired_network_"+str(i)+str(j)+"_safe.nnet"
        temp = NNet(nn_art_path_refine)
        biases = [np.array([bia]).T for bia in temp.biases]
        art_safe_model_refine = cp.deepcopy(our_safe_model)
        layer_indx = 0
        for layer in art_safe_model_refine:
            if isinstance(layer, torch.nn.Linear):
                layer.weight = torch.nn.Parameter(torch.tensor(temp.weights[layer_indx],dtype=torch.float32))
                layer.bias = torch.nn.Parameter(torch.tensor(biases[layer_indx][:,0],dtype=torch.float32))
                layer_indx += 1
        with torch.no_grad():
            art_refine_outputs = art_safe_model_refine(all_inputs)
        diff_art_refine_outputs = torch.mean((art_refine_outputs - orig_outputs).norm(dim=1, p=2))
        all_diff_art_refine_outputs.append(diff_art_refine_outputs)


        try:
            nn_art_path_no_refine = "art_test_goal_safety_no_refine/repaired_network_" + str(i) + str(j) + "_safe.nnet"
            temp = NNet(nn_art_path_no_refine)
        except:
            nn_art_path_no_refine = "art_test_goal_safety_no_refine/repaired_network_" + str(i) + str(
                j) + "_unsafe.nnet"
            temp = NNet(nn_art_path_no_refine)

        biases = [np.array([bia]).T for bia in temp.biases]
        art_safe_model_no_refine = cp.deepcopy(our_safe_model)
        layer_indx = 0
        for layer in art_safe_model_no_refine:
            if isinstance(layer, torch.nn.Linear):
                layer.weight = torch.nn.Parameter(torch.tensor(temp.weights[layer_indx],dtype=torch.float32))
                layer.bias = torch.nn.Parameter(torch.tensor(biases[layer_indx][:,0],dtype=torch.float32))
                layer_indx += 1
        with torch.no_grad():
            art_no_refine_outputs = art_safe_model_no_refine(all_inputs)
        diff_art_no_refine_outputs = torch.mean((art_no_refine_outputs - orig_outputs).norm(dim=1, p=2))
        all_diff_art_no_refine_outputs.append(diff_art_no_refine_outputs)

        sio.savemat('output_distance.mat', {'all_diff_our_outputs': all_diff_our_outputs,
                                            'all_diff_art_refine_outputs': all_diff_art_refine_outputs,
                                            'all_diff_art_no_refine_outputs': all_diff_art_no_refine_outputs})

    return all_diff_our_outputs, all_diff_art_refine_outputs, all_diff_art_no_refine_outputs



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


def compute_minimal_repair_expressibility():
    all_nn_info = []
    dir1 = 'logs_lrxxx_epochs200_alpha_beta/'
    for i in range(1,6):
        for j in range(1,10):
            net_best_accu = 0.0
            best_accu_dir = ''
            for lr in [0.01, 0.001]:
                for alpha_beta in [[0.2,0.8],[0.5,0.5],[0.8,0.2]]:
                    dir2 = 'nnet' + str(i) + str(j) + '_lr' + str(lr) + '_epochs200' + \
                               '_alpha' + str(alpha_beta[0]) + '_beta'+str(alpha_beta[1])+'/'
                    filename = 'all_test_accuracy.mat'
                    try:
                        dicts = sio.loadmat(dir1+dir2+filename)
                        accu = dicts['all_test_accuracy'][:,-1]
                        if accu > net_best_accu:
                            net_best_accu = accu
                            best_accu_dir = dir1+dir2
                    except:
                        continue
            if best_accu_dir != '':
                all_nn_info.append([best_accu_dir,[i,j]])


    if os.path.isfile('all_our_minimal_lregions.mat'):
       all_linear_regions =  sio.loadmat('all_our_minimal_lregions.mat')
       all_ori_lregions = list(all_linear_regions['all_ori_lregions'][0,:])
       all_our_lregions = list(all_linear_regions['all_our_lregions'][0,:])
       start_id = len(all_ori_lregions)
    else:
        all_ori_lregions = []
        all_our_lregions = []
        start_id = 0

    for indx, nn_dir_id in enumerate(all_nn_info):
        if indx < start_id:
            continue
        nn_dir = nn_dir_id[0]
        nn_id = nn_dir_id[1]

        i, j = nn_id[0], nn_id[1]
        print('Neural Network',i,j)
        properties_repair = None
        for item in repair_list:
            if item[0] == [i,j]:
                properties_repair = item[1]
                break
        assert properties_repair is not None

        nn_original_path = "../nets/ACASXU_run2a_" + str(i) + "_" + str(j) + "_batch_2000.onnx"
        original_model = load_ffnn_onnx(nn_original_path)
        ori_lregion = compute_linear_regions(original_model, properties_repair)
        all_ori_lregions.append(ori_lregion)

        list_of_files_path = glob.glob(nn_dir + '/*')
        # safe_net_path = max(list_of_files_path, key=os.path.getctime)
        for safe_net_path in list_of_files_path:
            if safe_net_path[-7:-3] == 'safe':
                break
        our_safe_model = torch.load(safe_net_path)
        our_lregion = compute_linear_regions(our_safe_model, properties_repair)
        all_our_lregions.append(our_lregion)

        sio.savemat('all_our_minimal_lregions.mat', {'all_our_lregions': all_our_lregions,
                                                 'all_ori_lregions': all_ori_lregions})


def compute_expressibility():
    if os.path.isfile('all_linear_regions.mat'):
       all_linear_regions =  sio.loadmat('all_linear_regions.mat')
       all_ori_lregions = list(all_linear_regions['all_ori_lregions'][0,:])
       all_our_lregions = list(all_linear_regions['all_our_lregions'][0,:])
       all_art_lregion_refines = list(all_linear_regions['all_art_lregion_refines'][0,:])
       all_art_lregion_no_refines = list(all_linear_regions['all_art_lregion_no_refines'][0,:])
       start_id = len(all_ori_lregions)
    else:
        all_ori_lregions = []
        all_our_lregions = []
        all_art_lregion_refines = []
        all_art_lregion_no_refines = []
        start_id = 0

    for indx, item in enumerate(repair_list):
        if indx == 0 or indx <= start_id:
            continue

        i, j = item[0][0], item[0][1]
        print('Neural Network',i,j)
        properties_repair = item[1]

        nn_original_path = "../nets/ACASXU_run2a_" + str(i) + "_" + str(j) + "_batch_2000.onnx"
        original_model = load_ffnn_onnx(nn_original_path)
        ori_lregion = compute_linear_regions(original_model, properties_repair)
        all_ori_lregions.append(ori_lregion)

        nn_our_path = "logs_lr0.001_epochs200/nnet" + str(i) + str(j) + "_lr0.001_epochs200"
        list_of_files_path = glob.glob(nn_our_path + '/*')
        # safe_net_path = max(list_of_files_path, key=os.path.getctime)
        for safe_net_path in list_of_files_path:
            if safe_net_path[-7:-3] == 'safe':
                break
        our_safe_model = torch.load(safe_net_path)
        our_lregion = compute_linear_regions(our_safe_model, properties_repair)
        all_our_lregions.append(our_lregion)

        nn_art_path_refine = "art_test_goal_safety/repaired_network_"+str(i)+str(j)+"_safe.nnet"
        art_safe_model_refine = NNet(nn_art_path_refine)
        art_lregion_refine = compute_linear_regions(art_safe_model_refine, properties_repair)
        all_art_lregion_refines.append(art_lregion_refine)

        try:
            nn_art_path_no_refine = "art_test_goal_safety_no_refine/repaired_network_" + str(i) + str(j) + "_safe.nnet"
            art_safe_model_no_refine = NNet(nn_art_path_no_refine)
        except:
            nn_art_path_no_refine = "art_test_goal_safety_no_refine/repaired_network_" + str(i) + str(j) + "_unsafe.nnet"
            art_safe_model_no_refine = NNet(nn_art_path_no_refine)

        art_lregion_no_refine = compute_linear_regions(art_safe_model_no_refine, properties_repair)
        all_art_lregion_no_refines.append(art_lregion_no_refine)

        sio.savemat('all_linear_regions.mat', {'all_ori_lregions': all_ori_lregions,
                                              'all_our_lregions': all_our_lregions,
                                              'all_art_lregion_refines': all_art_lregion_refines,
                                               'all_art_lregion_no_refines': all_art_lregion_no_refines})

    return all_ori_lregions, all_our_lregions, all_art_lregion_refines, all_art_lregion_no_refines


def find_weights_nnet_minimal_repair():
    all_nn_info = []
    dir1 = 'logs_lrxxx_epochs200_alpha_beta/'
    for i in range(1,6):
        for j in range(1,10):
            net_best_accu = 0.0
            best_accu_dir = ''
            for lr in [0.01, 0.001]:
                for alpha_beta in [[0.2,0.8],[0.5,0.5],[0.8,0.2]]:
                    dir2 = 'nnet' + str(i) + str(j) + '_lr' + str(lr) + '_epochs200' + \
                               '_alpha' + str(alpha_beta[0]) + '_beta'+str(alpha_beta[1])+'/'
                    filename = 'all_test_accuracy.mat'
                    try:
                        dicts = sio.loadmat(dir1+dir2+filename)
                        accu = dicts['all_test_accuracy'][:,-1]
                        if accu > net_best_accu:
                            net_best_accu = accu
                            best_accu_dir = dir1+dir2
                    except:
                        continue
            if best_accu_dir != '':
                all_nn_info.append([best_accu_dir,[i,j]])

    our_deviation_minimal_repair = []
    for nn_dir_id in all_nn_info:
        nn_dir = nn_dir_id[0]
        nn_id = nn_dir_id[1]
        list_of_files_path = glob.glob(nn_dir + '/*')
        # safe_net_path = max(list_of_files_path, key=os.path.getctime)
        for safe_net_path in list_of_files_path:
            if safe_net_path[-7:-3] == 'safe':
                break

        assert safe_net_path[-7:-3] == 'safe'
        our_safe_model = torch.load(safe_net_path)
        our_weights, our_bias = extract_weights(our_safe_model)

        i, j = nn_id[0], nn_id[1]
        nn_original_path = "../nets/ACASXU_run2a_" + str(i) + "_" + str(j) + "_batch_2000.onnx"
        original_model = load_ffnn_onnx(nn_original_path)
        original_weights, original_bias = extract_weights(original_model)

        square_sum_ours = 0
        square_sum_base = 0
        for layer in range(len(original_weights)):
            square_sum_base += np.sum(np.square(original_weights[layer])) + np.sum(np.square(original_bias[layer]))

            our_diff = np.sum(np.square(our_weights[layer] - original_weights[layer])) \
                       + np.sum(np.square(our_bias[layer] - original_bias[layer]))
            square_sum_ours += our_diff

        our_deviation_minimal_repair.append(np.sqrt(square_sum_ours)/np.sqrt(square_sum_base))

    return our_deviation_minimal_repair


def compute_weights_ratio():
    all_our_deviation = []
    all_art_refine = []
    all_art_no_refine = []
    for anet in repaired_nets:
        i, j = anet[0],anet[1]
        nn_our_path = "logs_lr0.001_epochs200/nnet" + str(i) + str(j) + "_lr0.001_epochs200"
        list_of_files_path = glob.glob(nn_our_path+'/*')
        # safe_net_path = max(list_of_files_path, key=os.path.getctime)
        for safe_net_path in list_of_files_path:
            if safe_net_path[-7:-3] == 'safe':
                break
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

        all_our_deviation.append(our_deviation)
        all_art_refine.append(art_refine)
        all_art_no_refine.append(art_no_refine)

    all_minimal_repair_deviation = find_weights_nnet_minimal_repair()

    return all_minimal_repair_deviation, all_our_deviation, all_art_refine, all_art_no_refine


if __name__ == "__main__":

    # results_refine, results_no_refine = collect_accuracy_time()
    # sio.savemat('art_accuracy_time.mat', {'results_refine': results_refine,
    #                                       'results_no_refine': results_no_refine})
    all_our_deviation, all_minimal_repair_deviation, all_art_refine, all_art_no_refine = compute_weights_ratio()
    sio.savemat('weights_deviation.mat', {'all_our_deviation': all_our_deviation,
                                          'all_minimal_repair_deviation': all_minimal_repair_deviation,
                                          'all_art_refine_dev': all_art_refine,
                                          'all_art_no_refine_dev': all_art_no_refine})

    # all_ori_lregions, all_our_lregions, all_art_lregion_refines, all_art_lregion_no_refines = compute_expressibility()
    # sio.savemat('all_linear_regions.mat', {'all_ori_lregions': all_ori_lregions,
    #                                       'all_our_lregions': all_our_lregions,
    #                                       'all_art_lregion_refines': all_art_lregion_refines,
    #                                        'all_art_lregion_no_refines': all_art_lregion_no_refines})
    # compute_minimal_repair_expressibility()

    # all_diff_our_outputs, all_diff_art_refine_outputs, all_diff_art_no_refine_outputs = compute_reachable_set_distance()
    # sio.savemat('output_distance.mat', {'all_diff_our_outputs': all_diff_our_outputs,
    #                                       'all_diff_art_refine_outputs': all_diff_art_refine_outputs,
    #                                       'all_diff_art_no_refine_outputs': all_diff_art_no_refine_outputs})



