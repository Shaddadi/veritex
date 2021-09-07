import sys
sys.path.insert(0, '../../src')
import copy as cp
from scipy.io import loadmat
from nnet_work_steal import DNN
from acasxu_properties import *
import multiprocessing as mp
from worker import Worker
from shared import SharedState
import time


if __name__ == "__main__":
    all_times = []
    all_results = []
    #
    # for i in range(1,6):
    #     for j in range(1,10):
    # i, j = 4, 6
    # while True:
    i, j = 1, 9
    nn_path = "nnet-mat-files/ACASXU_run2a_" + str(i) + "_" + str(j) + "_batch_2000.mat"
    filemat = loadmat(nn_path)
    W = filemat['W'][0]
    b = filemat['b'][0]

    properties = [property7]
    dnn0 = DNN(W, b)
    for n, p in enumerate(properties):
        t0 = time.time()
        vfl_input = cp.deepcopy(p.input_set)
        dnn0.unsafe_domains = p.unsafe_domains

        num_processors = 20
        processes = []
        shared_state = SharedState([vfl_input], num_processors, dnn0._num_layer)
        one_worker = Worker(dnn0)
        for index in range(num_processors):
            p = mp.Process(target=one_worker.main_func, args=(index, shared_state))
            processes.append(p)
            p.start()

        # for p in processes:
        #     p.terminate()

        while not shared_state.outputs.empty():
            _ = shared_state.outputs.get()

        for p in processes:
            p.join()

        print('Network '+str(i)+str(j)+ ' on property'+str(n+1))
        print('Running Time: ', time.time()-t0)



