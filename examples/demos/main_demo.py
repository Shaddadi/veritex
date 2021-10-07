import sys
sys.path.insert(0, '../../src')
import copy as cp
from scipy.io import loadmat, savemat
from nnet import DNN
from methods import Methods
import numpy as np
from sfproperty import Property
import time


if __name__ == "__main__":
    nn_path = "nets/NeuralNetwork7_3.mat"
    filemat = loadmat(nn_path)
    W = filemat['W'][0]
    b = filemat['b'][0]

    dnn0 = DNN(W, b)
    dnn0.config_unsafe_input = True
    dnn0.config_exact_output = True

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
        meth = Methods(dnn0, [property1])
        results = meth.nnReach(unsafe_input=True)
        output_sets = [item[2][1] for sublist in results for item in sublist]
        unsafe_sets = [item[2][0][0] for sublist in results for item in sublist if item[2][0]]
        t = time.time() - t0
        all_time.append(t)

        out_sets = []
        for item in output_sets:
            out_vertices = np.dot(item.vertices, item.M.T) + item.b.T
            out_sets.append(out_vertices)

        all_out_sets.append(out_sets)

        unsafe_inputs = []
        for item in unsafe_sets:
            unsafe_inputs.append(item.vertices)

        all_unsafe_sets.append(unsafe_inputs)

        unsafe_domain_vs = np.array([[y1_lbs,-15], [y1_lbs, 25], [y1_ubs,-15], [y1_ubs, 25]])
        all_unsafe_domain.append(unsafe_domain_vs)

    savemat('inputs_outputs_sets.mat', {'all_out_sets':all_out_sets, 'all_unsafe_sets':all_unsafe_sets, 'all_unsafe_domain': all_unsafe_domain, 'all_time': all_time})


