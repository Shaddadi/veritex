import sys
sys.path.insert(0, '../../src')
import torch
from acasxu_properties import *
import multiprocessing as mp
from worker import Worker
from shared import SharedState
from load_onnx import load_ffnn_onnx
import multiprocessing
from repair import REPAIR, DATA
import torch.optim as optim
import torch.nn as nn
import os



if __name__ == '__main__':
    num_processors = multiprocessing.cpu_count()
    print('num_processors: ', num_processors)
    properties = [property1, property2, property3, property4]
    # properties = [property2]

    i, j = 1, 2
    nn_path = "nets/ACASXU_run2a_" + str(i) + "_" + str(j) + "_batch_2000.onnx"
    torch_model = load_ffnn_onnx(nn_path)

    # learning parameters for repairing
    lr = 0.00005
    epochs = 200

    rp = REPAIR(torch_model, properties)
    optimizer = optim.SGD(torch_model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.MSELoss() #nn.CrossEntropyLoss()
    savepath = './logs'
    if not os.path.isdir(savepath):
        os.mkdir(savepath)
    savepath += '/nnet'+str(i)+str(j)+'_lr'+str(lr)+'_epochs'+str(epochs)
    if not os.path.isdir(savepath):
        os.mkdir(savepath)
    rp.repair_model(torch_model, optimizer, criterion, savepath, epochs=epochs)


