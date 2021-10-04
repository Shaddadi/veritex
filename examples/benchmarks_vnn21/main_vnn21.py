import sys
sys.path.insert(0, '../../src')
# sys.path.insert(0, './src')
from cnn import CNN
import torch
import torch.nn
from load_onnx import load_model_onnx
import numpy as np
import onnxruntime as ort
from read_vnnlib import read_vnnlib_simple
from onnx import numpy_helper
import time



def check_model(modelpath, pytorch_model):
    ort_session = ort.InferenceSession(modelpath)
    outputs_orig = ort_session.run(None, {'input.1': np.ones((1, 3, 32, 32)).astype(np.float32)})

    pytorch_model.eval()
    outputs_convt = pytorch_model(torch.ones(1, 3, 32, 32, dtype=torch.float32))
    assert np.sum(np.abs(outputs_orig[0]-outputs_convt.numpy())) <= 1.0e-5


def load_cifar_image(imagepath):
    vnnlib = read_vnnlib_simple(imagepath, 3072, 10)
    label = np.argmax(vnnlib[0][1][0][0])
    image_range = torch.tensor(vnnlib[0][0])
    vnnlib_shape = (1, 3, 32, 32)
    image_lbs = image_range[:, 0].reshape(vnnlib_shape)
    image_ubs = image_range[:, 1].reshape(vnnlib_shape)

    return [image_lbs, image_ubs, label]



if __name__ == "__main__":
    model_type = 'CIFAR'
    if model_type == 'CIFAR':
        modelpath = './cifar2020/nets/cifar10_2_255.onnx'
        imagepath = './cifar2020/specs/cifar10/cifar10_spec_idx_2_eps_0.00784_n1.vnnlib'
        pytorch_model, is_channel_last = load_model_onnx(modelpath, input_shape=(3,32,32))
        check_model(modelpath, pytorch_model)
        inputs = load_cifar_image(imagepath)

        t0 = time.time()
        net_cnn = CNN(pytorch_model, is_cuda=False)
        xx = net_cnn.reach_over_appr_parallel(inputs)
        print('Time: ', time.time() - t0)
        # torch.multiprocessing.set_start_method('spawn',force=True)
        # # t0 = time.time()
        # result = net_cnn.reach_over_appr(inputs)
        # if not result: # unknown
        #     all_inputs = net_cnn.split_input([inputs], num=5)
        #
        # num_core = 5
        # pool = torch.multiprocessing.Pool(num_core)
        # results = pool.map(net_cnn.reach_over_appr_parallel, all_inputs)
        # xx = 1









