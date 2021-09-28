import sys
sys.path.insert(0, '../../src')
# sys.path.insert(0, './src')
from cnn import CNN
from vzono_cnn import Vzono
import torch
import torch.nn
from load_onnx import load_model_onnx
import numpy as np
import onnxruntime as ort
from read_vnnlib import read_vnnlib_simple
from onnx import numpy_helper



def check_model(modelpath, pytorch_model):
    ort_session = ort.InferenceSession(modelpath)
    outputs_orig = ort_session.run(None, {'input.1': np.ones((1, 3, 32, 32)).astype(np.float32)})

    pytorch_model.eval()
    outputs_convt = pytorch_model(torch.ones(1, 3, 32, 32, dtype=torch.float32))
    assert np.sum(np.abs(outputs_orig[0]-outputs_convt.numpy())) <= 1.0e-5


if __name__ == "__main__":
    model_type = 'CIFAR'
    if model_type == 'CIFAR':
        modelpath = './cifar2020/nets/cifar10_2_255.onnx'
        imagepath = './cifar2020/specs/cifar10/cifar10_spec_idx_0_eps_0.00784_n1.vnnlib'
        pytorch_model, is_channel_last = load_model_onnx(modelpath, input_shape=(3,32,32))
        check_model(modelpath, pytorch_model)

        vnnlib = read_vnnlib_simple(imagepath , 3072, 10)
        x_range = torch.tensor(vnnlib[0][0])
        vnnlib_shape = (1, 3, 32, 32)
        image_lbs = x_range[:, 0].reshape(vnnlib_shape)
        image_ubs = x_range[:, 1].reshape(vnnlib_shape)

        set_vzono = Vzono(image_lbs, image_ubs)
        xx = 1








