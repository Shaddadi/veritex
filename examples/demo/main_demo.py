import sys
sys.path.insert(0, '../../src')
import copy as cp
from scipy.io import loadmat
from ffnn_work_steal import FFNN
from worker import Worker
from shared import SharedState
import multiprocessing as mp
import numpy as np
from sfproperty import Property


if __name__ == "__main__":
    nn_path = "nets/NeuralNetwork7_3.mat"
    filemat = loadmat(nn_path)
    W = filemat['W'][0]
    b = filemat['b'][0]

    dnn0 = FFNN(W, b)
    dnn0.config_unsafe_input = True
    dnn0.config_exact_output = True


    lbs = [-1, -1, -1]
    ubs = [1, 1, 1]
    input_domain = [lbs, ubs]
    A_unsafe = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])
    d_unsafe = np.array([[-50], [40], [-15], [-25]])
    unsafe_domains = [[A_unsafe,d_unsafe]]
    property1 = Property(input_domain, unsafe_domains)

    dnn0.unsafe_domains = property1.unsafe_domains
    vfl_input = cp.deepcopy(property1.input_set)

    processes = []
    num_processors = 5
    shared_state = SharedState([vfl_input], num_processors)
    one_worker = Worker(dnn0)
    for index in range(num_processors):
        p = mp.Process(target=one_worker.main_func, args=(index, shared_state))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    while not shared_state.outputs.empty():
        unsafe = shared_state.outputs.get()


    xx = 1