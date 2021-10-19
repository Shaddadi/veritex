import sys
sys.path.insert(0, '../../src')
import copy as cp
from scipy.io import loadmat, savemat
from ffnn_work_steal import FFNN
import multiprocessing as mp
from worker import Worker
from shared import SharedState
import time
import numpy as np
from sfproperty import Property


if __name__ == "__main__":
    num_processors = 2 # mp.cpu_count()
    print('num_processors: ', num_processors)
    nn_path = "nets/NeuralNetwork7_3.mat"
    filemat = loadmat(nn_path)
    W = filemat['W'][0]
    b = filemat['b'][0]

    dnn0 = FFNN(W, b, unsafe_inputs=True, exact_output=True)

    all_unsafe_domain = []
    all_out_sets = []
    all_unsafe_sets = []
    all_time = []
    for n in range(55):
        lbs = [-1, -1, -1]
        ubs = [1, 1, 1]
        input_domain = [lbs, ubs]
        y1_lbs = -50 + n
        y1_ubs = -40 + n
        A_unsafe = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])
        d_unsafe = np.array([[y1_lbs], [-y1_ubs], [-15], [-25]])
        unsafe_domains = [[A_unsafe,d_unsafe]]
        property1 = Property(input_domain, unsafe_domains)

        t0 = time.time()
        results = []
        vfl_input = cp.deepcopy(property1.input_set)
        dnn0.unsafe_domains = property1.unsafe_domains

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

        print('results len: ', len(results))
        print('shared_outputs len: ', shared_state.outputs_len.value)
        print('')
    #     out_sets = []
    #     for item in output_sets:
    #         out_vertices = np.dot(item.vertices, item.M.T) + item.b.T
    #         out_sets.append(out_vertices)
    #
    #     all_out_sets.append(out_sets)
    #
    #     unsafe_inputs = []
    #     for item in unsafe_sets:
    #         unsafe_inputs.append(item.vertices)
    #
    #     all_unsafe_sets.append(unsafe_inputs)
    #
    #     unsafe_domain_vs = np.array([[y1_lbs,-15], [y1_lbs, 25], [y1_ubs,-15], [y1_ubs, 25]])
    #     all_unsafe_domain.append(unsafe_domain_vs)
    #
    # savemat('inputs_outputs_sets.mat', {'all_out_sets':all_out_sets, 'all_unsafe_sets':all_unsafe_sets, 'all_unsafe_domain': all_unsafe_domain, 'all_time': all_time})
    #
