import sys
import numpy as np
from veritex.utils.load_onnx import load_ffnn_onnx
# from acasxu_repair_list import *
import scipy.io as sio
from veritex.utils.load_nnet import NNet
import copy as cp
from veritex.networks.ffnn import FFNN
import multiprocessing as mp
from veritex.methods.worker import Worker
from veritex.methods.shared import SharedState
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



def get_log_info_art(log_path):
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
        safety = []
        for indx, l in enumerate(lines):
            if l[32:45] == 'For AcasNetID':
                net_IDs.append((int(l[46]), int(l[48])))
            elif l[32:55] == 'Accuracy at every epoch':
                accus = np.fromstring(l[58:-1],sep=',')
                accurcys.append(accus[-1])
                t = get_time(lines[indx+1])
                run_times.append(t)
                if lines[indx+2][78:] == 'True':
                    safety.append(True)
                elif lines[indx+2][78:] == 'False':
                    safety.append(False)


        assert len(net_IDs)==len(accurcys) and len(net_IDs)==len(run_times) and len(net_IDs)==len(safety)
        return [net_IDs, accurcys, run_times, safety]



def get_log_info_veritex(log_path):
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
        safety = []
        for indx, l in enumerate(lines):
            if l[31:45] == 'Neural Network':
                net_IDs.append((int(l[46]), int(l[48])))
            elif l[31:62] == 'The accurate and safe candidate':
                if l[79:] == 'True':
                    safety.append(True)
                elif l[79:] == 'False':
                    safety.append(False)

                accus = np.fromstring(lines[indx-2][58:-1],sep=',')
                accurcys.append(accus[-1])
                t = float(lines[indx+1].split(" ")[7])
                run_times.append(t)

        assert len(net_IDs)==len(accurcys) and len(net_IDs)==len(run_times) and len(net_IDs)==len(safety)
        return [net_IDs, accurcys, run_times, safety]


def collect_art_accuracy_runtime():
    art_log_path_refine = glob.glob("ART/results/acas/art_test_goal_safety/*.log")[-1]
    art_log_path_no_refine = glob.glob("ART/results/acas/art_test_goal_safety_no_refine/*.log")[-1]

    results_refine = get_log_info_art(art_log_path_refine)
    results_no_refine = get_log_info_art(art_log_path_no_refine)
    return results_refine, results_no_refine

def collect_veritex_accuracy_runtime():
    log_path = glob.glob("../../examples/ACASXu/repair/logs/*.log")[-1]
    results = get_log_info_veritex(log_path)
    return results

def collect_veritex_accuracy_runtime_minimal():
    log_path = glob.glob("../../examples/ACASXu/repair/logs_minimal/*.log")[-1]
    results = get_log_info_veritex(log_path)
    return results


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
    results_refine_art, results_no_refine_art = collect_art_accuracy_runtime()
    results_veritex = collect_veritex_accuracy_runtime()
    results_veritex_minimal = collect_veritex_accuracy_runtime_minimal()
    if not os.path.isdir('./results'):
        os.mkdir('./results')
    tables = ''
    if len(results_veritex[0]) == 33: # only simple cases
        for ls in results_refine_art:
            ls.pop(0)
            ls.pop(8)
        for ls in results_no_refine_art:
            ls.pop(0)
            ls.pop(8)

        tables += 'Table 1: Repair of ACAS Xu neural network controllers.\n'
        tables += '------------------------------------------------------------------------------------\n'
        tables += 'Methods                  Repair Success      Min Accu.    Mean Accu.     Max Accu.\n'
        tables += '------------------------------------------------------------------------------------\n'
        tables += f'ART                         {len(np.nonzero(results_no_refine_art[3])[0])}/33             {min(results_no_refine_art[1])*100:.2f}%       {np.mean(results_no_refine_art[1])*100:.2f}%         {max(results_no_refine_art[1])*100:.2f}%\n'
        tables += f'ART-refinement              {len(np.nonzero(results_refine_art[3])[0])}/33             {min(results_refine_art[1])*100:.2f}%       {np.mean(results_refine_art[1])*100:.2f}%         {max(results_refine_art[1])*100:.2f}%\n'
        tables += f'Our Non-minimal Repair      {len(np.nonzero(results_veritex[3])[0])}/33             {min(results_veritex[1]):.2f}%       {np.mean(results_veritex[1]):.2f}%         {max(results_veritex[1]):.2f}%\n'
        tables += f'Our Minimal Repair          {len(np.nonzero(results_veritex_minimal[3])[0])}/33             {min(results_veritex_minimal[1]):.2f}%       {np.mean(results_veritex_minimal[1]):.2f}%         {max(results_veritex_minimal[1]):.2f}%\n'
        tables += '------------------------------------------------------------------------------------\n'
        tables += '\n'
        tables += '\n'
        tables += 'Table 2: Running time (sec) of our repair method and ART\n'
        tables += '--------------------------------------------------------------------------\n'
        tables += 'Methods          Min     Mean Time     Max      Time (N19)     Time (N29)\n'
        tables += '--------------------------------------------------------------------------\n'
        tables += f'ART             {min(results_no_refine_art[2]):.2f}      {np.mean(results_no_refine_art[2]):.2f}      {max(results_no_refine_art[2]):.2f}      --            -- \n'
        tables += f'ART-refinement  {min(results_refine_art[2]):.2f}      {np.mean(results_refine_art[2]):.2f}      {max(results_refine_art[2]):.2f}       --            -- \n'
        tables += f'Our Method      {min(results_veritex[2]):.2f}       {np.mean(results_veritex[2]):.2f}      {max(results_veritex[2]):.2f}      --            -- \n'
        tables += '--------------------------------------------------------------------------\n'

    elif len(results_veritex[0]) == 35: # all cases
        tables += 'Table 1: Repair of ACAS Xu neural network controllers.\n'
        tables += '------------------------------------------------------------------------------------\n'
        tables += 'Methods                  Repair Success      Min Accu.    Mean Accu.     Max Accu.\n'
        tables += '------------------------------------------------------------------------------------\n'
        tables += f'ART                         {len(np.nonzero(results_no_refine_art[3])[0])}/33             {min(results_no_refine_art[1])*100:.2f}%       {np.mean(results_no_refine_art[1])*100:.2f}%         {max(results_no_refine_art[1])*100:.2f}%\n'
        tables += f'ART-refinement              {len(np.nonzero(results_refine_art[3])[0])}/33             {min(results_refine_art[1])*100:.2f}%       {np.mean(results_refine_art[1])*100:.2f}%         {max(results_refine_art[1])*100:.2f}%\n'
        tables += f'Our Non-minimal Repair      {len(np.nonzero(results_veritex[3])[0])}/33             {min(results_veritex[1]):.2f}%       {np.mean(results_veritex[1]):.2f}%         {max(results_veritex[1]):.2f}%\n'
        tables += f'Our Minimal Repair          {len(np.nonzero(results_veritex_minimal[3])[0])}/33             {min(results_veritex_minimal[1]):.2f}%       {np.mean(results_veritex_minimal[1]):.2f}%         {max(results_veritex_minimal[1]):.2f}%\n'
        tables += '------------------------------------------------------------------------------------\n'
        tables += '\n'
        tables += '\n'
        tables += 'Table 2: Running time (sec) of our repair method and ART\n'
        tables += '--------------------------------------------------------------------------\n'
        tables += 'Methods          Min     Mean Time     Max      Time (N19)     Time (N29)\n'
        tables += '--------------------------------------------------------------------------\n'
        tables += f'ART             {min(results_no_refine_art[2]):.2f}      {np.mean(results_no_refine_art[2]):.2f}      {max(results_no_refine_art[2]):.2f}      {results_no_refine_art[2][0]:.2f}            {results_no_refine_art[2][9]:.2f}\n'
        tables += f'ART-refinement  {min(results_refine_art[2]):.2f}      {np.mean(results_refine_art[2]):.2f}      {max(results_refine_art[2]):.2f}       {results_refine_art[2][0]:.2f}            {results_refine_art[2][9]:.2f}\n'
        tables += f'Our Method      {min(results_veritex[2]):.2f}       {np.mean(results_veritex[2]):.2f}      {max(results_veritex[2]):.2f}      {results_veritex[2][0]:.2f}            {results_veritex[2][9]:.2f}\n'
        tables += '--------------------------------------------------------------------------\n'

    else:
        sys.exit("The number of repaired instances is not correct!")

    with open('results/Table2&3.txt', 'w') as f:
        f.write(tables)
    print(tables)










