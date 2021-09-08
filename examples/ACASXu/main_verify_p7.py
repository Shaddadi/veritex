import sys
sys.path.insert(0, '../../src')
import copy as cp
from scipy.io import loadmat
from nnet_work_steal import DNN
from acasxu_properties import *
import multiprocessing as mp
from worker import Worker
from shared import SharedState
import multiprocessing
import time
import pickle


if __name__ == "__main__":
    all_times = []
    all_results = []

    num_processors = multiprocessing.cpu_count()
    print('num_processors: ', num_processors)
    properties = [property7]
    for n, prop in enumerate(properties):
        i, j = 1, 9
        nn_path = "nnet-mat-files/ACASXU_run2a_" + str(i) + "_" + str(j) + "_batch_2000.mat"
        filemat = loadmat(nn_path)
        W = filemat['W'][0]
        b = filemat['b'][0]

        dnn0 = DNN(W, b)
        dnn0.config_verify = True
        dnn0.config_relu_linear = True

        t0 = time.time()
        unsafe = False
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
            unsafe = shared_state.outputs.get()

        print('')
        print('Network '+str(i)+str(j)+ ' on property 7')
        print('Unsafe: ', unsafe)
        print('Running Time: ', time.time()-t0)
        all_times.append(time.time()-t0)
        all_results.append(unsafe)

    with open('verification_p7.pkl', 'wb') as f:
        pickle.dump([all_times, all_results], f)





