import sys
sys.path.insert(0, '../../src')
import copy as cp
from scipy.io import loadmat
from ffnn import FFNN
from acasxu_properties import *
import multiprocessing as mp
from logs.nnet_file import NNet
from worker import Worker
from shared import SharedState
from load_onnx import load_ffnn_onnx
from scipy.io import loadmat, savemat
from acasxu_repair_list import *
import multiprocessing
import time
import pickle


if __name__ == "__main__":
    all_times = []
    all_results = []

    num_processors = multiprocessing.cpu_count()
    print('num_processors: ', num_processors)

    item = repair_list[1]
    i, j = item[0][0], item[0][1]
    # properties = [item[1][1]]
    properties = [item[1][indx] for indx in range(2,4)]
    t0 = time.time()
    for iter in range(1):
        nn_path = "logs/logs_lr0.001_epochs200/nnet21_lr0.001_epochs200/acasxu_epoch24_safe.pt"
        torch_model = torch.load(nn_path)
        # nn_path = "logs/art_test_goal_safety/repaired_network_21_safe.nnet"
        # model = NNet(nn_path)
        # biases = [np.array([bia]).T for bia in model.biases]
        # torch_model = [model.weights, biases]
        # nn_path = "nets/ACASXU_run2a_" + str(i) + "_" + str(j) + "_batch_2000.onnx"
        # torch_model = load_ffnn_onnx(nn_path)

        all_out_sets = []
        all_out_unsafe_sets = []
        for n, prop in enumerate(properties):
            dnn0 = FFNN(torch_model, unsafe_inputs=True, exact_output=True)

            results = []
            vfl_input = cp.deepcopy(prop[0].input_set)
            dnn0.unsafe_domains = prop[0].unsafe_domains

            processes = []
            shared_state = SharedState([vfl_input], num_processors)
            one_worker = Worker(dnn0)
            for index in range(num_processors):
                p = mp.Process(target=one_worker.main_func, args=(index, shared_state))
                processes.append(p)
                p.start()

            for p in processes:
                p.join()

            while not shared_state.outputs.empty():
                results.append(shared_state.outputs.get())

            output_sets = [item[1] for item in results]
            unsafe_sets = []
            for item in results:
                if item[0]:
                    unsafe_sets.extend(item[0])

            for item in output_sets:
                out_vertices = np.dot(item.vertices, item.M.T) + item.b.T
                all_out_sets.append(out_vertices)

            out_unsafe_sets = []
            for item in unsafe_sets:
                out_unsafe_vertices = np.dot(item.vertices, item.M.T) + item.b.T
                all_out_unsafe_sets.append(out_unsafe_vertices)


        # savemat('logs/nnet12_lr2e-05_epochs200/outputs_sets_p2'+str(iter)+'.mat', {'all_out_sets': all_out_sets, 'all_out_unsafe_sets': all_out_unsafe_sets})
        savemat('logs/nnet21/outputs_sets_p34_our.mat',
                {'all_out_sets': all_out_sets, 'all_out_unsafe_sets': all_out_unsafe_sets})

    print('Time: ', time.time() - t0)

