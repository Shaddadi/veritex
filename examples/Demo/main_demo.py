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

# plot function
def plot_sets(input_unsafe_sets, output_sets, unsafe_domain, savepath='', image_id=0):
    fig = plt.figure(figsize=(11, 5))
    ax = fig.add_subplot(121, projection='3d')
    fig.suptitle('Instance '+str(image_id), fontsize=18)
    for set_vs in input_unsafe_sets:
        plot_polytope3d(set_vs, ax, color='r', alpha=1, edgecolor='k',linewidth=0.2)

    input_lbs, input_ubs = [-1, -1, -1], [1, 1, 1]
    plot_box3d(input_lbs, input_ubs, ax, color='b', alpha=0.2, edgecolor='k', linewidth=1.0)

    ax.dist = 10
    ax.azim = -125
    ax.elev = 25
    ax.set_xlim3d(-1.1,1.1)
    ax.set_ylim3d(-1.1,1.1)
    ax.set_zlim3d(-1.1,1.1)
    ax.set_xlabel('$x_1$', fontsize=15)
    ax.set_ylabel('$x_2$', fontsize=15)
    ax.set_zlabel('$x_3$',fontsize=15)
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    ax.set_zticks([-1, 0, 1])
    ax.title.set_text('Input domain & Exact unsafe subspace')

    ax2 = fig.add_subplot(122)
    for indx, set_vs in enumerate(output_sets):
        plot_polytope2d(set_vs, ax2, color='b',alpha=1.0, edgecolor='k',linewidth=0.0)

    plot_polytope2d(unsafe_domain, ax2, color='r', alpha=0.5, edgecolor='k', linewidth=0.4)
    ax2.set_xlim(-54, 28)
    ax2.set_ylim(-18, 28)
    ax2.set_xlabel('$y_1$', fontsize=15)
    ax2.set_ylabel('$y_2$', fontsize=15)
    ax2.title.set_text('Exact output reachable domain & Unsafe domain')
    plt.savefig(savepath+'image'+str(image_id)+'.png')
    plt.close()


if __name__ == "__main__":
    # Creating and Configuring Logger
    logger = logging.getLogger()
    for hdlr in logger.handlers[:]:  # remove all old handlers
        logger.removeHandler(hdlr)
    Log_Format = logging.Formatter('%(levelname)s %(asctime)s - %(message)s')
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler('demo.log','w+')
    file_handler.setFormatter(Log_Format)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(Log_Format)
    logger.addHandler(console_handler)

    # init FFNN
    logging.info('Start the demo...')
    model = torch.load('demo_model.pt')
    dnn0 = FFNN(model, unsafe_in_dom=True, exact_out_dom=True)

    all_unsafe_domain = []
    all_output_sets = []
    all_input_unsafe_sets = []
    for n in range(55):
        logging.info(f'Completed instances: {n+1}/55')

        # set property (= pre-condition and post-condition)
        lbs = [-1, -1, -1]
        ubs = [1, 1, 1]
        input_domain = [lbs, ubs]
        y1_lbs = -50 + n
        y1_ubs = -40 + n
        A_unsafe = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])
        d_unsafe = np.array([[y1_lbs], [-y1_ubs], [-15], [-25]])
        unsafe_domains = [[A_unsafe,d_unsafe]]
        property1 = Property(input_domain, unsafe_domains)
        dnn0.unsafe_domains = property1.unsafe_domains

        # run reachability analysis with multi-process
        processes = []
        num_processors = mp.cpu_count()
        vfl_input = cp.deepcopy(property1.input_set)
        shared_state = SharedState([vfl_input], num_processors)
        one_worker = Worker(dnn0)
        for index in range(num_processors):
            p = mp.Process(target=one_worker.main_func, args=(index, shared_state))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()

        results = []
        while not shared_state.outputs.empty():
            results.append(shared_state.outputs.get())

        # get results
        output_sets = [np.dot(item[1].vertices, item[1].M.T) + item[1].b.T for item in results]  # sets represented with their vertices
        input_unsafe_sets = [sub.vertices for item in results for sub in item[0] if sub]  # sets represented with their vertices
        unsafe_domain = np.array([[y1_lbs, -15], [y1_lbs, 25], [y1_ubs, -15], [y1_ubs, 25]])

        # plot
        if not os.path.exists('images'):
            os.makedirs('images')
        plot_sets(input_unsafe_sets, output_sets, unsafe_domain, savepath='images/', image_id=n)

    logging.info('The input-output reachable domain of each instance is printed in /images')