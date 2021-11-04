import onnx
import onnxruntime as ort
from onnx import numpy_helper
import torch
from torch import nn
import numpy as np
import copy
# pip install onnx2pytorch
from onnx2pytorch import ConvertModel






filename = 'cifar10_2_255.onnx'
onnx_model = onnx.load(filename)
onnx.checker.check_model(onnx_model)
graph = onnx_model.graph

pytorch_model = ConvertModel(onnx_model)
pytorch_model.eval()
yy = pytorch_model(torch.ones(1, 3, 32, 32, dtype=torch.float32))

net_layers = []
for layer in pytorch_model.children():
    if type(layer).__name__ in ['Conv2d', 'Linear', 'ReLU', 'Reshape']:
        if type(layer).__name__ == 'Reshape':
            net_layers.append(nn.Flatten())
        else:
            net_layers.append(layer)

sequential_layers = nn.Sequential(*net_layers).eval()
for param in sequential_layers.parameters():
    param.requires_grad = False

im = torch.ones(1, 3, 32, 32, dtype=torch.float32)
all_inputs = [im.numpy()]
for i in range(len(sequential_layers)):
    im = sequential_layers[i](im, bias=False)
    all_inputs.append(copy.deepcopy(im.numpy()))

xx = 1

# output_seq = sequential_layers(torch.ones(1, 3, 32, 32, dtype=torch.float32))
# ort_session = ort.InferenceSession(filename)
# outputs = ort_session.run(None, {'input.1': np.ones((1, 3, 32, 32)).astype(np.float32)})
# # print(outputs[0])
