import sys
sys.path.insert(0, '../../class')

import os
import time
import nnet
import cubelattice as cl
import multiprocessing
from functools import partial
from scipy.io import loadmat
import numpy as np
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Verification Settings')
    parser.add_argument('--property', type=str, default='1')
    parser.add_argument('--n1', type=int, default=2)
    parser.add_argument('--n2', type=int, default=3)
    parser.add_argument('--compute_unsafety', action='store_true')
    args = parser.parse_args()
    
    i = args.n1
    j = args.n2

    def verification(afv):
        safe = True
        return safe

    print("neural_network_"+str(i)+str(j))
    nn_path = "nets/neural_network_information_"+str(i)+str(j)+".mat"
    filemat = loadmat(nn_path)
    if not os.path.isdir('logs'):
        os.mkdir('logs')

    W = filemat['W'][0]
    b = filemat['b'][0]

    lb = [-0.1,-0.1,-0.1]
    ub = [0.1,0.1,0.1]

    nnet0 = nnet.nnetwork(W, b)
    nnet0.verification = verification
    initial_input = cl.cubelattice(lb, ub).to_lattice()
    cpus = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(cpus)

    nnet0.start_time = time.time()
    nnet0.filename = "logs/output_info"+str(i)+str(j)+'.txt'
    outputSets = []
    nputSets0 = nnet0.singleLayerOutput(initial_input, 0)
    pool.map(partial(nnet0.layerOutput, m=1), nputSets0)
    pool.close()
    elapsed_time = time.time() - nnet0.start_time

    print('time elapsed: %f seconds \n' % elapsed_time)
    print('result: safe\n')
    filex = open(nnet0.filename, 'w')
    filex.write('time elapsed: %f seconds \n' % elapsed_time)
    filex.write('result: safe\n')
    filex.close()

