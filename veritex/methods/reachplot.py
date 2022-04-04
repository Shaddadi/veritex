import sys
import os.path
import copy as cp
import multiprocessing as mp
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing
import argparse
from veritex.utils.load_nnet import NNet

from veritex.networks.ffnn import FFNN
from veritex.utils.plot_poly import plot_polytope2d
from veritex.methods.worker import Worker
from veritex.methods.shared import SharedState
from veritex.utils.load_onnx import load_ffnn_onnx
from examples.ACASXu.repair.acasxu_properties import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plotting of reachable domains')
    parser.add_argument('--property', type=str, required=False)
    parser.add_argument('--network_path', type=str, required=False)
    parser.add_argument('--savename', type=str, required=False)
    parser.add_argument('--dims', nargs='+', type=int, default=(0, 1))
    args = parser.parse_args()
    prop_indx = np.fromstring(args.property, dtype=int, sep=',')
    props = []
    for n in prop_indx:
        prop_name = 'property' + str(n)
        assert prop_name in all_properties
        props.append(all_properties[prop_name])
    assert props

    network_path = args.network_path
    dim0, dim1 = args.dims

    try:
        if network_path[-4:] == 'onnx':
            torch_model = load_ffnn_onnx(network_path)
        elif network_path[-2:] == 'pt':
            torch_model = torch.load(network_path)
        elif network_path[-4:] == 'nnet':
            model = NNet(network_path)
            biases = [np.array([bia]).T for bia in model.biases]
            torch_model = [model.weights, biases]
        else:
            torch_model = None
    except:
        sys.exit('Network file is not found!')

    fig = plt.figure(figsize=(2.0, 2.67))
    ax = fig.add_subplot(111)
    dnn0 = FFNN(torch_model, unsafe_in_dom=True, exact_out_dom=True)
    for prop in props:
        vfl_input = cp.deepcopy(prop.input_set)
        dnn0.unsafe_domains = prop.unsafe_domains

        processes = []
        results = []
        num_processors = multiprocessing.cpu_count()
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

        output_sets = [item[1] for item in results]
        unsafe_sets = []
        for item in results:
            if item[0]: unsafe_sets.extend(item[0])

        for item in output_sets:
            out_vertices = np.dot(item.vertices, item.M.T) + item.b.T
            plot_polytope2d(out_vertices[:, [dim0, dim1]], ax, color='b', alpha=1.0, edgecolor='k', linewidth=0.0)

        all_output_unsafe_sets = []
        for item in unsafe_sets:
            out_unsafe_vertices = np.dot(item.vertices, item.M.T) + item.b.T
            plot_polytope2d(out_unsafe_vertices[:, [dim0, dim1]], ax, color='r', alpha=1.0, edgecolor='k',
                            linewidth=0.0)

    ax.autoscale()
    ax.set_xlabel('$y_' + str(dim0) + '$', fontsize=16)
    ax.set_ylabel('$y_' + str(dim1) + '$', fontsize=16)
    #plt.title('Exact output reachable domain (blue) & Unsafe domain (red) on'+' Property '+args.property, fontsize=18, pad=20)

    plt.savefig(args.savename + '.png', bbox_inches='tight')
    plt.close()






