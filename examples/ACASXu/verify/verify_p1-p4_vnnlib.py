import sys
import copy as cp
from veritex.networks.ffnn import FFNN
import multiprocessing as mp
from veritex.methods.worker import Worker
from veritex.methods.shared import SharedState
from veritex.utils.load_onnx import load_ffnn_onnx
from veritex.utils.vnnlib import vnnlib_to_properties
import multiprocessing
import logging
import torch
import time
import pickle



if __name__ == "__main__":
    all_times = []
    all_results = []
    # Creating and Configuring Logger
    logger = logging.getLogger()
    Log_Format = logging.Formatter('%(levelname)s %(asctime)s - %(message)s')
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler('verify_p1_p4.log', 'w+')
    file_handler.setFormatter(Log_Format)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(Log_Format)
    logger.addHandler(console_handler)

    num_processors = multiprocessing.cpu_count()
    print('num_processors: ', num_processors)

    # extract safety properties from vnnlib files
    properties = []
    for n in range(1, 5):
        prop = vnnlib_to_properties('../nets/prop_' + str(n) + '.vnnlib', num_inputs=5, num_outputs=5)
        properties.extend(prop)

    # run safety verification
    for n, prop in enumerate(properties):
        for i in range(1,6):
            for j in range(1,10):
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
                logging.info(f'Network {i} {j} on property {n+1}')
                logging.info(f'Unsafe: {unsafe}')
                logging.info(f'Running Time: {time.time() - t0} sec')
                all_times.append(time.time()-t0)
                all_results.append(unsafe)

    # with open('verification_p1_p4.pkl', 'wb') as f:
    #     pickle.dump([all_times, all_results], f)





