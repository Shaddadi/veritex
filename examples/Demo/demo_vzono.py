import os.path
import logging
import copy as cp
import numpy as np
import torch
import multiprocessing as mp
import matplotlib.pyplot as plt

from veritex.networks.ffnn import FFNN
from veritex.methods.worker import Worker
from veritex.methods.shared import SharedState
from veritex.utils.sfproperty import Property
from veritex.utils.plot_poly import *

# get current directory
currdir = os.path.dirname(os.path.abspath(__file__))

def reach_relu_network():
    """
    Load ReLU network and conduct reachability analysis of the network based on FVIM set representation.
    """
    model = torch.load(f'{currdir}/models/model_relu.pt')
    dnn0 = FFNN(model, exact_outputd=True)

    # Set property
    lbs, ubs = [-0.1, -0.1, -0.1], [0.1, 0.1, 0.1]
    property1 = Property([lbs, ubs], [], set_type='FVIM')
    dnn0.set_property(property1)

    # Run reachability analysis with the work-stealing parallel
    processes = []
    num_processors = mp.cpu_count()
    shared_state = SharedState(property1, num_processors)
    one_worker = Worker(dnn0)
    for index in range(num_processors):
        p = mp.Process(target=one_worker.main_func, args=(index, shared_state))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    outputs = []
    while not shared_state.outputs.empty():
        outputs.append(shared_state.outputs.get())

    # Extract vertices of output reachable sets
    exact_output_sets = [np.dot(item.vertices, item.M.T) + item.b.T for item in outputs]

    # Over approximate output reachable domain withe Vzono set representation
    property1 = Property([lbs, ubs], [], set_type='Vzono')
    dnn0.set_property(property1)
    appr_domain = dnn0.reach_over_approximation()
    appr_vs = appr_domain.get_vertices()

    # Simulation
    inputs = []
    num = 1000
    for i in range(len(lbs)):
        inputs.append(np.random.uniform(lbs[i], ubs[i], num))

    inputs = np.array(inputs)
    outputs = dnn0.simulate(inputs)

    # Plot output reachable domain and simulations
    fig = plt.figure()
    ax = fig.add_subplot(111)
    dim0, dim1 = 0, 1
    plot_polytope2d(appr_vs[[dim0, dim1],:].T, ax, color='c', alpha=1.0, edgecolor='k', linewidth=0.0, zorder=1)
    for vs in exact_output_sets:
        plot_polytope2d(vs[:, [dim0, dim1]], ax, color='b', alpha=1.0, edgecolor='k', linewidth=0.0,zorder=2)

    ax.plot(outputs[dim0,:], outputs[dim1,:],'k.', markersize=1, zorder=3)
    ax.autoscale()
    ax.set_xlabel(f'$y_{dim0}$', fontsize=16)
    ax.set_ylabel(f'$y_{dim1}$', fontsize=16)
    plt.title('Reachable domain of a ReLU network.\nCyan area represents the over-approximated reachable domain. \n Blue area represents the exact reachable domain. \nBlack dots represent simultations')
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'{currdir}/figures/relu_vzono.png')
    plt.close()



def reach_sigmoid_network():
    """
    Load Sigmoid network and conduct exact reachability analysis based on Vzono set representation
    """

    model = torch.load(f'{currdir}/models/model_sigmoid.pt')
    dnn0 = FFNN(model)

    # Set property
    lbs, ubs = [-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]
    property1 = Property([lbs, ubs], [], set_type='Vzono')
    dnn0.set_property(property1)
    appr_domain = dnn0.reach_over_approximation()
    # appr_vs = appr_domain.get_vertices() # there will be too many vertices generated
    appr_vs = appr_domain.get_sound_vertices()

    # Simulation
    inputs = []
    num = 5000
    for i in range(len(lbs)):
        inputs.append(np.random.uniform(lbs[i], ubs[i], num))

    inputs = np.array(inputs)
    outputs = dnn0.simulate(inputs)

    # Plot output reachable domain and simulations
    fig = plt.figure()
    ax = fig.add_subplot(111)
    dim0, dim1 = 0, 1
    plot_polytope2d(appr_vs[[dim0, dim1], :].T, ax, color='c', alpha=1.0, edgecolor='k', linewidth=0.0, zorder=1)
    ax.plot(outputs[dim0, :], outputs[dim1, :], 'k.', markersize=1, zorder=2)
    ax.autoscale()
    ax.set_xlabel(f'$y_{dim0}$', fontsize=16)
    ax.set_ylabel(f'$y_{dim1}$', fontsize=16)
    plt.title(
        'Reachable domain of a Sigmoid network.\nCyan area represents the over-approximated reachable domain.\nBlack dots represent simultations')
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'{currdir}/figures/sigmoid_vzono.png')
    plt.close()



def reach_tanh_network():
    """
    Load Tanh network and conduct exact reachability analysis based on Vzono set representation
    """

    model = torch.load(f'{currdir}/models/model_tanh.pt')
    dnn0 = FFNN(model)
    # Set property
    lbs, ubs = [-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]
    property1 = Property([lbs, ubs], [], set_type='Vzono')
    dnn0.set_property(property1)
    appr_domain = dnn0.reach_over_approximation()
    # appr_vs = appr_domain.get_vertices() # there will be too many vertices generated
    appr_vs = appr_domain.get_sound_vertices()

    # Simulation
    inputs = []
    num = 5000
    for i in range(len(lbs)):
        inputs.append(np.random.uniform(lbs[i], ubs[i], num))

    inputs = np.array(inputs)
    outputs = dnn0.simulate(inputs)

    # Plot output reachable domain and simulations
    fig = plt.figure()
    ax = fig.add_subplot(111)
    dim0, dim1 = 0, 1
    plot_polytope2d(appr_vs[[dim0, dim1], :].T, ax, color='c', alpha=1.0, edgecolor='k', linewidth=0.0, zorder=1)
    ax.plot(outputs[dim0, :], outputs[dim1, :], 'k.', markersize=1, zorder=2)
    ax.autoscale()
    ax.set_xlabel(f'$y_{dim0}$', fontsize=16)
    ax.set_ylabel(f'$y_{dim1}$', fontsize=16)
    plt.title(
        'Reachable domain of a Tanh network.\nCyan area represents the over-approximated reachable domain.\nBlack dots represent simultations')
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'{currdir}/figures/tanh_vzono.png')
    plt.close()



if __name__ == "__main__":
    reach_relu_network()
    reach_sigmoid_network()
    reach_tanh_network()







