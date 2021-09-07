import sys
sys.path.insert(0, '../../src')

from scipy.io import loadmat
from nnet import DNN
from methods import Methods
from acasxu_properties import *
import time



if __name__ == "__main__":
    all_times = []
    all_results = []

    # for i in range(1,6):
    #     for j in range(1,10):
    i, j = 1, 9
    nn_path = "nnet-mat-files/ACASXU_run2a_" + str(i) + "_" + str(j) + "_batch_2000.mat"
    filemat = loadmat(nn_path)
    W = filemat['W'][0]
    b = filemat['b'][0]

    t0 = time.time()
    properties = [property7]
    dnn0 = DNN(W, b)
    meth = Methods(dnn0, properties)
    verification = meth.verify(relu_linear=True)
    p_result = np.any(np.array(verification))
    print('Safety property 1 on Network: N'+str(i)+str(j))
    print('Unsafe: ', p_result)
    print('Running time(sec): %.2f' % (time.time() - t0))
    print('\n')
    # all_times.append(time.time() - t0)
    # all_results.append(p_result)

    # results = meth.nnReach(exact_output=True)
    # print('Safety property 1 on Network: N'+str(i)+str(j))
    # print('Output sets: ', len(results))
    # print('Running time(sec): %.2f' % (time.time() - t0))
    # print('\n')
