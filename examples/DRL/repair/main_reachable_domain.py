import os.path
import sys
sys.path.insert(0, '../../../src')
import copy as cp
from veritex.networks.ffnn import FFNN
from agent_properties import *
from veritex.utils.plot_poly import plot_polytope2d
import multiprocessing as mp
from veritex.methods.worker import Worker
from veritex.methods.shared import SharedState
from veritex.utils.load_onnx import load_ffnn_onnx
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing
import argparse
import torch

# get current directory
currdir = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Verification Settings')
    parser.add_argument('--property', type=str, required=False, help='It supports 0, 1')
    parser.add_argument('--network_path', type=str, required=False)
    parser.add_argument('--dims',  nargs='+', type=int, default=(0,1))
    args = parser.parse_args()
    args.property = '1, 2'
    prop_indx = np.fromstring(args.property, dtype=int, sep=',')
    props = []
    for n in prop_indx:
        prop_name = 'property' + str(n)
        assert prop_name in all_properties
        props.append(all_properties[prop_name])
    assert props
    network_path = f'{currdir}/../nets/unsafe_agent0.pt' #args.network_path
    # network_path = 'logs/agent2_lr1e-06_epochs50_alpha1.0_beta0.0/repaired_model.pt'
    dim0, dim1 = (0,1) #tuple(args.dims)

    if network_path[-4:]=='onnx':
        torch_model = load_ffnn_onnx(network_path)
    elif network_path[-2:]=='pt':
        torch_model = torch.load(network_path)
    else:
        sys.exit('Network file is not found!')

    fig = plt.figure(figsize=(2, 2.67))
    ax = fig.add_subplot(111)
    dnn0 = FFNN(torch_model, unsafe_inputd=True, exact_outputd=True)
    for prop in props:
        # vfl_input = cp.deepcopy(prop.input_set)
        vfl = cp.deepcopy(prop)
        dnn0.unsafe_domains = prop.unsafe_domains

        processes = []
        results = []
        num_processors = multiprocessing.cpu_count()
        # shared_state = SharedState(vfl_input, num_processors)
        shared_state = SharedState(vfl, num_processors)
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
            plot_polytope2d(out_vertices[:,[dim0,dim1]], ax, color='b',alpha=1.0, edgecolor='k',linewidth=0.0)

        all_output_unsafe_sets = []
        for item in unsafe_sets:
            out_unsafe_vertices = np.dot(item.vertices, item.M.T) + item.b.T
            plot_polytope2d(out_unsafe_vertices[:,[dim0,dim1]], ax, color='r', alpha=1.0, edgecolor='k', linewidth=0.0)

    ax.autoscale()
    ax.set_xlabel('$y_'+str(dim0)+'$', fontsize=16)
    ax.set_ylabel('$y_'+str(dim1)+'$', fontsize=16)
    # plt.title('Exact output reachable domain (blue) & Unsafe domain (red) on'+' Property '+args.property, fontsize=18, pad=20)
    if not os.path.exists(f'{currdir}/images'):
        os.mkdir(f'{currdir}/images')

    plt.savefig(f'{currdir}/images/reachable_domain_'+'property_'+args.property+'_dims'+str(dim0)+'_'+str(dim1)+'.png',bbox_inches='tight')
    plt.close()







