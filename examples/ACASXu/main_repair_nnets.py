import sys
sys.path.insert(0, '../../src')
from acasxu_repair_list import *
from load_onnx import load_ffnn_onnx
import multiprocessing
from utils import split_bounds,split_bounds_vset
from repair import REPAIR, DATA
import torch.optim as optim
import torch.nn as nn
import time
import os



if __name__ == '__main__':
    num_processors = multiprocessing.cpu_count()
    print('num_processors: ', num_processors)



    # for n in range(5, 40):
    # for n in range(40):
    n = 5
    lr = 0.005 #0.05 # learning parameters for repairing
    epochs = 200
    item = repair_list[n]
    i, j = item[0][0], item[0][1]
    properties = item[1]
    # nn_path = "nets/ACASXU_run2a_" + str(i) + "_" + str(j) + "_batch_2000.onnx"
    # torch_model = load_ffnn_onnx(nn_path)
    torch_model = torch.load('logs/nnet19_lr0.005_epochs200/acasxu_epoch1.pt')

    # rp = REPAIR(torch_model, properties, output_limit=1)
    # optimizer = optim.SGD(torch_model.parameters(), lr=lr, momentum=0.9)
    # criterion = nn.MSELoss() #nn.CrossEntropyLoss()
    # savepath = './logs'
    # if not os.path.isdir(savepath):
    #     os.mkdir(savepath)
    # savepath += '/nnet'+str(i)+str(j)+'_lr'+str(lr)+'_epochs'+str(epochs)
    # if not os.path.isdir(savepath):
    #     os.mkdir(savepath)
    # # savepath += '/repairing_lr'+str(lr)+'_epochs'+str(epochs)
    # # if not os.path.isdir(savepath):
    # #     os.mkdir(savepath)
    # # savepath += '/nnet'+str(i)+str(j)
    # # if not os.path.isdir(savepath):
    # #     os.mkdir(savepath)
    # # else:
    # #     continue
    # t0 = time.time()
    # rp.repair_model(optimizer, criterion, savepath, epochs=epochs)
    # print('Time :', time.time() - t0)

    # rp = REPAIR(torch_model, properties)
    # unsafe_data = rp.compute_unsafety()
    # print(unsafe_data)
    # print(np.all([len(sub) == 0 for sub in unsafe_data]))
    # rp.fast_verify_layers()

    # over approximation with vzono
    from ffnn import FFNN
    import copy as cp
    for prop in properties:
        nnet = FFNN(torch_model, verify=True)
        nnet.unsafe_domains = prop.unsafe_domains
        lbs_input, ubs_input = prop.lbs, prop.ubs
        sub_vzono_sets = split_bounds(lbs_input, ubs_input, num=4)
        num_cores = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(num_cores)
        results = []
        for item in sub_vzono_sets:
            nnet.split_verify_vzono_depth_first(item)
        # results.extend(pool.imap(nnet.split_verify_vzono_depth_first, sub_vzono_sets))
        xx = 1

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




