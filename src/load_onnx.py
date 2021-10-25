import onnx
import onnxruntime as ort
from onnx import numpy_helper
import gzip
import torch
from torch import nn
import onnx2pytorch  # pip install onnx2pytorch


def load_ffnn_onnx(path):
    onnx_model = onnx.load(path)
    pytorch_model = onnx2pytorch.ConvertModel(onnx_model)

    modules = list(pytorch_model.modules())[1:]
    new_modules = []
    for m in modules:
        if isinstance(m, torch.nn.ReLU) or isinstance(m, torch.nn.Linear):
            new_modules.append(m)

    torch_model = nn.Sequential(*new_modules)
    for param in torch_model.parameters():
        param.requires_grad = False

    return torch_model