
import time
import nnet
import cubelattice as cl
import multiprocessing
from functools import partial
from scipy.linalg import null_space
import copy as cp
import torch
import numpy as np
import random
import psutil

def compute_unsafety(model, datax, over_app=False):
    W = []
    b = []
    for name, param in model.named_parameters():
        if name[-4:] == 'ight':
            if torch.cuda.is_available():
                W.append(param.data.cpu().numpy())
            else:
                W.append(param.data.numpy())
        if name[-4:] == 'bias':
            if torch.cuda.is_available():
                temp = np.expand_dims(param.data.cpu().numpy(), axis=1)
                b.append(temp)
            else:
                temp = np.expand_dims(param.data.numpy(), axis=1)
                b.append(temp)

    # start_time = time.time()
    all_unsafe_data = []
    all_unsafe_sets = []
    all_sets = []
    t0 = time.time()
    largest_used_memory = 0
    for n, pty in enumerate(datax.properties):
        cpus = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(cpus)
        lb = pty[0][0]
        ub = pty[0][1]
        nnet0 = nnet.nnetwork(W, b, pty)
        initial_input = cl.cubelattice(lb, ub).to_lattice()

        outputSets = []
        nputSets0 = nnet0.singleLayerOutput(initial_input, 0)
        outputSets.extend(pool.imap(partial(nnet0.layerOutput, layer=1, over_app=over_app), nputSets0))

        unsafe_data = [item[0] for sublist in outputSets for item in sublist]
        unsafe_data_real = [item for item in unsafe_data if item is not None]

        unsafe_sets = [item[1] for sublist in outputSets for item in sublist]
        unsafe_sets_real = [np.matmul(item.vertices, item.M.T)+item.b.T for item in unsafe_sets if item is not None]

        allsets = [item[2] for sublist in outputSets for item in sublist]
        all_sets_real = [ np.matmul(item.vertices, item.M.T)+item.b.T for item in allsets if item is not None]


        all_unsafe_data.append(unsafe_data_real)
        all_unsafe_sets.extend(unsafe_sets_real)
        all_sets.extend(all_sets_real)

        used_memory = psutil.virtual_memory()[3]
        if used_memory>largest_used_memory:
            largest_used_memory = used_memory

        pool.close()

    return all_unsafe_data, time.time()-t0, largest_used_memory, all_unsafe_sets, all_sets


def check_over_approximation(nnet0, initial_input, lbs, ubs):
    num = 10000
    random_inputs = []
    for i in range(len(lbs)):
        lb = lbs[i]
        ub = ubs[i]
        random_inputs.append(np.random.uniform(lb,ub,(1,num)))

    random_inputs = np.concatenate(random_inputs, axis=0)
    outputs = nnet0.outputPoint(random_inputs.T)
    over_app = nnet0.layerOutputOverApp(initial_input, 0)

    outputs = outputs.T
    outputs_max = np.max(outputs, axis=1, keepdims=True)
    outputs_min = np.min(outputs, axis=1, keepdims=True)

    val = np.sum(np.abs(over_app.base_vectors), axis=1, keepdims=True)
    over_app_max = over_app.base_vertices + val
    over_app_min = over_app.base_vertices - val

    xx = over_app_max - outputs_max # should be all positive
    yy = over_app_min - outputs_min # should be all negative
    zz = 1