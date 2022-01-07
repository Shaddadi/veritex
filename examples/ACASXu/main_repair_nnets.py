import sys

import numpy as np

sys.path.insert(0, '../../src')
from acasxu_repair_list import *
from load_onnx import load_ffnn_onnx, save_onnx
import multiprocessing
from utils import split_bounds,split_bounds_abs
from repair import REPAIR, DATA
import torch.optim as optim
import torch.nn as nn
import time
import os



if __name__ == '__main__':
    num_processors = multiprocessing.cpu_count()
    print('num_processors: ', num_processors)



    # for n in range(5, 40):
    for n in range(1,len(repair_list)):
    # n = 0
        lr = 0.001 #0.05 # learning parameters for repairing
        epochs = 200
        alpha, beta = 0.8, 0.2
        item = repair_list[n]
        i, j = item[0][0], item[0][1]
        print('Neural Network',i,j)
        properties_repair = item[1]
        nn_path = "nets/ACASXU_run2a_" + str(i) + "_" + str(j) + "_batch_2000.onnx"
        torch_model = load_ffnn_onnx(nn_path)
        # torch_model = torch.load('logs/nnet29_lr0.001_epochs200/acasxu_epoch12_safe.pt')
        # save_onnx(torch_model, 5, 'acasxu_nnet29_epoch12_safe.onnx')
        # torch_model = torch.load('model_safe_for_xiaodong.pt')

        rp = REPAIR(torch_model, properties_repair, output_limit=1000)
        # rp = REPAIR(torch_model, properties_repair, output_limit=np.Inf)
        optimizer = optim.SGD(torch_model.parameters(), lr=lr, momentum=0.9)
        criterion = nn.MSELoss() #nn.CrossEntropyLoss()
        savepath = './logs'
        if not os.path.isdir(savepath):
            os.mkdir(savepath)
        savepath += '/nnet'+str(i)+str(j)+'_lr'+str(lr)+'_epochs'+str(epochs)+'_alpha'+str(alpha)+'_beta'+str(beta)
        if not os.path.isdir(savepath):
            os.mkdir(savepath)

        rp.repair_model(optimizer, criterion, alpha, beta, savepath, epochs=epochs)



    # # over approximation with diffabs
    # from utils import absmodel_from_torch, test_model, split_bounds_abs2
    # from diffabs.deeppoly import Dom
    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # # dom = Dom()
    # # input_bounds = torch.tensor([[
    # #     [-1., 1.],
    # #     [-1., 1.]
    # # ]], device=device)
    # #
    # # lb = input_bounds[:, :, 0]
    # # ub = input_bounds[:, :, 1]
    # # input = dom.Ele.by_intvl(lb, ub)
    # # absmodel = test_model()
    # # out = absmodel(input)
    # # lb = out.lb()
    # # ub = out.ub()
    # # xx = 1
    # torch_model = torch.load('model_safe_for_xiaodong.pt')
    # prop = properties_repair[4][0]
    # absmodel = absmodel_from_torch(torch_model)
    # lbs_input, ubs_input = prop.lbs, prop.ubs
    # input = split_bounds_abs2(lbs_input, ubs_input)
    # out = absmodel(input)
    # lb = out.lb()
    # ub = out.ub()
    # xx = 1




    # # over approximation with vzono
    # from ffnn import FFNN
    # import copy as cp
    # t0 = time.time()
    # for item in properties_repair:
    #     prop = item[0]
    #     nnet = FFNN(torch_model, verify=True)
    #     nnet.unsafe_domains = prop.unsafe_domains
    #     lbs_input, ubs_input = prop.lbs, prop.ubs
    #     sub_vzono_sets = split_bounds(lbs_input, ubs_input, num=3)
    #     num_cores = multiprocessing.cpu_count()
    #     pool = multiprocessing.Pool(num_cores)
    #     results = []
    #     # for item in sub_vzono_sets:
    #     #     nnet.split_verify_vzono_depth_first(item)
    #     results.extend(pool.imap(nnet.split_verify_vzono_depth_first, sub_vzono_sets))
    #
    # print('Time: ', time.time()-t0)

    # # over approximation with vset
    # from ffnn import FFNN
    # import copy as cp
    #
    # for prop in properties:
    #     nnet = FFNN(torch_model, verify=True)
    #     nnet.unsafe_domains = prop.unsafe_domains
    #     lbs_input, ubs_input = prop.lbs, prop.ubs
    #     sub_vzono_sets = split_bounds(lbs_input, ubs_input, num=2)
    #     for item in sub_vzono_sets:
    #         nnet.split_verify_vset_depth_first(item)




