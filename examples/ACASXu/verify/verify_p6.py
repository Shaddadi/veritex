
import copy as cp
from scipy.io import loadmat
from veritex.networks.ffnn import FFNN
from acasxu_properties import *
import multiprocessing as mp
from veritex.methods.worker import Worker
from veritex.methods.shared import SharedState
from veritex.utils.load_onnx import load_ffnn_onnx
import multiprocessing
import time
import pickle
import logging
import pickle


if __name__ == "__main__":
    all_times = []
    all_results = []
    # Creating and Configuring Logger
    logger = logging.getLogger()
    Log_Format = logging.Formatter('%(levelname)s %(asctime)s - %(message)s')
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler('verify_p6.log', 'w+')
    file_handler.setFormatter(Log_Format)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(Log_Format)
    logger.addHandler(console_handler)

    num_processors = multiprocessing.cpu_count()
    print('num_processors: ', num_processors)
    properties = [property6_1, property6_2]
    for n, prop in enumerate(properties):
        i, j = 1, 1
        nn_path = "../nets/ACASXU_run2a_" + str(i) + "_" + str(j) + "_batch_2000.onnx"
        torch_sequential = load_ffnn_onnx(nn_path)
        dnn0 = FFNN(torch_sequential, verify=True, relu_linear=True)

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

        logging.info('')
        logging.info(f'Network {i}{j} on property 6')
        logging.info(f'Unsafe: {unsafe}')
        logging.info(f'Running Time: {time.time() - t0} sec')
        all_times.append(time.time()-t0)
        all_results.append(unsafe)

    with open('verification_p6.pkl', 'wb') as f:
        pickle.dump([all_times, all_results], f)





