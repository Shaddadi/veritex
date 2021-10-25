import sys
sys.path.insert(0, '../../src')
import copy as cp
from scipy.io import loadmat
from ffnn import FFNN
from acasxu_properties import *
import multiprocessing as mp
from worker import Worker
from shared import SharedState
from load_onnx import load_ffnn_onnx
import multiprocessing
import time
import pickle


if __name__ == "__main__":
    all_times = []
    all_results = []

    num_processors = multiprocessing.cpu_count()
    print('num_processors: ', num_processors)
    properties = [property2]
    t0 = time.time()
    for n, prop in enumerate(properties):
        i, j = 1, 2
        nn_path = "nets/ACASXU_run2a_" + str(i) + "_" + str(j) + "_batch_2000.onnx"
        torch_sequential = load_ffnn_onnx(nn_path)

        dnn0 = FFNN(torch_sequential, unsafe_inputs=True, exact_output=True)

        t0 = time.time()
        results = []
        vfl_input = cp.deepcopy(prop.input_set)
        dnn0.unsafe_domains = prop.unsafe_domains

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


    print('Time: ', time.time() - t0)

