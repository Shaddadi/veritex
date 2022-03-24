import sys
import logging
sys.path.insert(0, '../../../src')
from agent_repair_list import *
from veritex.utils.load_onnx import load_ffnn_onnx, save_onnx
import multiprocessing
from veritex.methods.repair import REPAIR, DATA
import torch.optim as optim
import torch.nn as nn
import torch
import os



if __name__ == '__main__':
    # Creating and Configuring Logger
    logger = logging.getLogger()
    Log_Format = logging.Formatter('%(levelname)s %(asctime)s - %(message)s')
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler('neural_network_repair.log')
    file_handler.setFormatter(Log_Format)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(Log_Format)
    logger.addHandler(console_handler)

    num_processors = multiprocessing.cpu_count()
    for n in range(2,len(repair_list)):
        lr = 0.000001
        epochs = 50
        alpha, beta = 1.0, 0.0
        nnet_id = repair_list[n][0]
        properties_repair = repair_list[n][1]
        nn_path = "../nets/unsafe_agent"+str(nnet_id)+".pt"
        torch_model = torch.load(nn_path).to(torch.float32)

        rp = REPAIR(torch_model, properties_repair, output_limit=1000)
        optimizer = optim.SGD(torch_model.parameters(), lr=lr, momentum=0.9)
        criterion = nn.MSELoss()
        savepath = './logs'
        if not os.path.isdir(savepath):
            os.mkdir(savepath)
        savepath += '/agent'+str(n)+'_lr'+str(lr)+'_epochs'+str(epochs)+'_alpha'+str(alpha)+'_beta'+str(beta)
        if not os.path.isdir(savepath):
            os.mkdir(savepath)

        rp.repair_model_regular(optimizer, criterion, alpha, beta, savepath, epochs=epochs)



