import sys
sys.path.insert(0, '../../src')
from acasxu_repair_list import *
from load_onnx import load_ffnn_onnx
import multiprocessing
from repair import REPAIR, DATA
import torch.optim as optim
import torch.nn as nn
import time
import os



if __name__ == '__main__':
    num_processors = multiprocessing.cpu_count()
    print('num_processors: ', num_processors)



    # for item in repair_list:
    n = 4
    lr = 0.00005 # learning parameters for repairing
    epochs = 200
    item = repair_list[n]
    i, j = item[0][0], item[0][1]
    properties = item[1]
    nn_path = "nets/ACASXU_run2a_" + str(i) + "_" + str(j) + "_batch_2000.onnx"
    torch_model = load_ffnn_onnx(nn_path)
    # torch_model = torch.load('logs/nnet16_lr2e-05_epochs200/acasxu_epoch2_safe.pt')

    rp = REPAIR(torch_model, properties)
    optimizer = optim.SGD(torch_model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.MSELoss() #nn.CrossEntropyLoss()
    savepath = './logs'
    if not os.path.isdir(savepath):
        os.mkdir(savepath)
    savepath += '/nnet'+str(i)+str(j)+'_lr'+str(lr)+'_epochs'+str(epochs)
    if not os.path.isdir(savepath):
        os.mkdir(savepath)
    # savepath += '/repairing_lr'+str(lr)+'_epochs'+str(epochs)
    # if not os.path.isdir(savepath):
    #     os.mkdir(savepath)
    # savepath += '/nnet'+str(i)+str(j)
    # if not os.path.isdir(savepath):
    #     os.mkdir(savepath)
    # else:
    #     continue
    t0 = time.time()
    rp.repair_model(optimizer, criterion, savepath, epochs=epochs)
    print('Time :', time.time() - t0)


