import os
import logging
import multiprocessing
import torch.optim as optim
import torch.nn as nn
from veritex.methods.repair import REPAIR, DATA
from veritex.utils.load_onnx import load_ffnn_onnx, save_onnx
from acasxu_repair_list import *
import argparse

# get current directory
currdir = os.path.dirname(os.path.abspath(__file__))

if __name__ == '__main__':
    savepath = f'{currdir}/logs_minimal'
    if not os.path.isdir(savepath):
        os.mkdir(savepath)

    # Creating and Configuring Logger
    logger = logging.getLogger()
    Log_Format = logging.Formatter('%(levelname)s %(asctime)s - %(message)s')
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(savepath+'/neural_network_repair_minimal.log', 'w+')
    file_handler.setFormatter(Log_Format)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(Log_Format)
    logger.addHandler(console_handler)

    parser = argparse.ArgumentParser(description='Plotting of reachable domains')
    parser.add_argument('--all', action='store_true')
    args = parser.parse_args()
    if args.all:
        logging.info(f'Repair all unsafe networks')
        networks = repair_list
    else:
        networks = repair_list[:33]

    num_processors = multiprocessing.cpu_count()
    for n in range(len(networks)):
        lr = 0.001
        epochs = 200
        alpha, beta = 0.8, 0.2
        item = networks[n]
        i, j = item[0][0], item[0][1]
        if (i==1 and j ==9) or (i==2 and j ==9):
            output_limit = 10
            lr = 0.01
        else:
            output_limit = 100
        logging.info(f'Neural Network {i} {j}')
        properties_repair = item[1]
        nn_path = f"{currdir}/../nets/ACASXU_run2a_" + str(i) + "_" + str(j) + "_batch_2000.onnx"
        torch_model = load_ffnn_onnx(nn_path)

        rp = REPAIR(torch_model, properties_repair, output_limit=output_limit)
        optimizer = optim.SGD(torch_model.parameters(), lr=lr, momentum=0.9)
        criterion = nn.MSELoss()

        filepath = savepath + '/nnet'+str(i)+str(j)+'_lr'+str(lr)+'_epochs'+str(epochs)+'_alpha'+str(alpha)+'_beta'+str(beta)
        if not os.path.isdir(filepath):
            os.mkdir(filepath)

        rp.repair_model_classification(optimizer, criterion, alpha, beta, filepath, epochs=epochs,iters=200)
        logging.info('\n****************************************************************\n')



