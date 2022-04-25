
import torch.multiprocessing
import matplotlib.pyplot as plt
from veritex.networks.cnn import Method
import data.cifar_torch_net as cifar10
from veritex.utils.plot_poly import plot_polytope2d
import logging
import numpy as np


classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn',force=True)
    # Creating and Configuring Logger
    logger = logging.getLogger()
    for hdlr in logger.handlers[:]:  # remove all old handlers
        logger.removeHandler(hdlr)
    Log_Format = logging.Formatter('%(levelname)s %(asctime)s - %(message)s')
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(f'cifar10.log', 'w+')
    file_handler.setFormatter(Log_Format)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(Log_Format)
    logger.addHandler(console_handler)

    # Load cifar10 model
    model = cifar10.Net()
    model.load_state_dict(torch.load('data/cifar_torch_net.pth', map_location=torch.device('cpu')))
    model.eval()

    # Load target image
    [image, label, target_label, _] = torch.load('data/images/3.pt')

    attack_block = (1,1)
    epsilon = 0.02
    relaxation = 0.6

    logging.info(f'Size of block: {attack_block}')
    logging.info(f'Perturbation epsilon: {epsilon}')
    logging.info(f'Neuron relaxation: {relaxation * 100} %')
    logging.info(f'Image label: {label}')

    reach_model = Method(model, image, label, 'logs',
                         attack_block=attack_block,
                         epsilon=epsilon,
                         relaxation=relaxation)
    output_sets = reach_model.reach()

    sims = reach_model.simulate()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    dim0, dim1 = label.numpy(), target_label.numpy()

    for item in output_sets:
        out_vertices = item[0]
        plot_polytope2d(out_vertices[:, [dim0, dim1]], ax, color='b', alpha=1.0, edgecolor='k', linewidth=0.0,zorder=2)

    ax.plot(sims[:,dim0], sims[:,dim1],'k.',zorder=1)
    ax.autoscale()
    ax.set_xlabel('$y_' + str(dim0) + '$', fontsize=16)
    ax.set_ylabel('$y_' + str(dim1) + '$', fontsize=16)
    plt.tight_layout()
    plt.show()
    plt.close()



