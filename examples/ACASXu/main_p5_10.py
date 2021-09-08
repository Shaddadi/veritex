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

    # i, j = 1,1
    # nn_path = "nnet-mat-files/ACASXU_run2a_" + str(i) + "_" + str(j) + "_batch_2000.mat"
    # filemat = loadmat(nn_path)
    # W = filemat['W'][0]
    # b = filemat['b'][0]
    #
    # t0 = time.time()
    # properties = [property5]
    # dnn0 = DNN(W, b)
    # meth = Methods(dnn0, properties)
    # verification = meth.verify(relu_linear=True)
    # p_result = np.any(np.array(verification))
    # print('Safety property: 5')
    # print('Unsafe: ', p_result)
    # print('Running time(sec): %.2f' % (time.time() - t0))
    # print('\n')
    # all_times.append(time.time() - t0)
    # all_results.append(p_result)
    #
    # t0 = time.time()
    # properties = [property6_1]
    # dnn0 = DNN(W, b)
    # meth = Methods(dnn0, properties)
    # verification = meth.verify(relu_linear=True)
    # p_result = np.any(np.array(verification))
    # print('Safety property: 6.1')
    # print('Unsafe: ', p_result)
    # print('Running time(sec): %.2f' % (time.time() - t0))
    # print('\n')
    # all_times.append(time.time() - t0)
    # all_results.append(p_result)
    #
    # t0 = time.time()
    # properties = [property6_2]
    # dnn0 = DNN(W, b)
    # meth = Methods(dnn0, properties)
    # verification = meth.verify(relu_linear=True)
    # p_result = np.any(np.array(verification))
    # print('Safety property: 6.2')
    # print('Unsafe: ', p_result)
    # print('Running time(sec): %.2f' % (time.time() - t0))
    # print('\n')
    # all_times.append(time.time() - t0)
    # all_results.append(p_result)
    #
    #
    # i, j = 1,9
    # nn_path = "nnet-mat-files/ACASXU_run2a_" + str(i) + "_" + str(j) + "_batch_2000.mat"
    # filemat = loadmat(nn_path)
    # W = filemat['W'][0]
    # b = filemat['b'][0]
    # properties = [property7]
    # dnn0 = DNN(W, b)
    # meth = Methods(dnn0, properties)
    # verification = meth.verify(relu_linear=True)
    # p_result = np.any(np.array(verification))
    # print('Safety property: 7')
    # print('Unsafe: ', p_result)
    # print('Running time(sec): %.2f' % (time.time() - t0))
    # print('\n')
    # all_times.append(time.time() - t0)
    # all_results.append(p_result)
    #

    i, j = 2,9
    t0 = time.time()
    nn_path = "nnet-mat-files/ACASXU_run2a_" + str(i) + "_" + str(j) + "_batch_2000.mat"
    filemat = loadmat(nn_path)
    W = filemat['W'][0]
    b = filemat['b'][0]
    properties = [property8]
    dnn0 = DNN(W, b)
    meth = Methods(dnn0, properties)
    verification = meth.verify(relu_linear=True)
    p_result = np.any(np.array(verification))
    print('Safety property: 8')
    print('Unsafe: ', p_result)
    print('Running time(sec): %.2f' % (time.time() - t0))
    print('\n')
    all_times.append(time.time() - t0)
    all_results.append(p_result)


    # i, j = 3,3
    # nn_path = "nnet-mat-files/ACASXU_run2a_" + str(i) + "_" + str(j) + "_batch_2000.mat"
    # filemat = loadmat(nn_path)
    # W = filemat['W'][0]
    # b = filemat['b'][0]
    # properties = [property9]
    # dnn0 = DNN(W, b)
    # meth = Methods(dnn0, properties)
    # verification = meth.verify(relu_linear=True)
    # p_result = np.any(np.array(verification))
    # print('Safety property: 9')
    # print('Unsafe: ', p_result)
    # print('Running time(sec): %.2f' % (time.time() - t0))
    # print('\n')
    # all_times.append(time.time() - t0)
    # all_results.append(p_result)
    #
    #
    # i, j = 4,5
    # nn_path = "nnet-mat-files/ACASXU_run2a_" + str(i) + "_" + str(j) + "_batch_2000.mat"
    # filemat = loadmat(nn_path)
    # W = filemat['W'][0]
    # b = filemat['b'][0]
    # properties = [property10]
    # dnn0 = DNN(W, b)
    # meth = Methods(dnn0, properties)
    # verification = meth.verify(relu_linear=True)
    # p_result = np.any(np.array(verification))
    # print('Safety property: 10')
    # print('Unsafe: ', p_result)
    # print('Running time(sec): %.2f' % (time.time() - t0))
    # print('\n')
    # all_times.append(time.time() - t0)
    # all_results.append(p_result)
    #
    #
