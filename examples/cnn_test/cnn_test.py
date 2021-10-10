import sys
sys.path.insert(0, '../../src')
# sys.path.insert(0, './src')
from cnn import CNN
import torch
import copy as cp
import torch.nn as nn
import pickle
import numpy as np
import scipy.io
import time

class Net(nn.Module):
    def __init__(self):
        """CNN Builder."""
        super(Net, self).__init__()

        self.sequential = nn.Sequential(
            nn.Linear(2, 3),
            nn.ReLU(inplace=True),
            nn.Linear(3, 3),
            nn.ReLU(inplace=True),
            nn.Linear(3, 3),
            nn.ReLU(inplace=True),
            nn.Linear(3, 2)
        )


    def forward(self, x):
        x = self.sequential(x)
        return x


def generate_vertices(vzono_set):
    all_vertices = [vzono_set.base_vertices]
    for v in vzono_set.base_vectors:
        temp = []
        for vertex in all_vertices:
            temp.append(vertex + v)
            temp.append(vertex - v)
        all_vertices = temp

    all_vertices= torch.cat(all_vertices, dim=0).numpy()
    return all_vertices


if __name__ == "__main__":
    model = Net()
    PATH = 'model_test.pt'
    # torch.save(model.state_dict(), PATH)
    model.load_state_dict(torch.load(PATH))
    model.eval()
    pytorch_model = model.sequential
    with torch.no_grad():
        pytorch_model[0].bias = torch.nn.Parameter(torch.tensor([0.0, 0.0, 0.0]))
        pytorch_model[2].bias = torch.nn.Parameter(torch.tensor([0.0, 0.0, 0.0]))
        pytorch_model[4].bias = torch.nn.Parameter(torch.tensor([0.0, 0.0, 0.0]))

    for param in pytorch_model.parameters():
        param.requires_grad = False

    lbs, ubs = torch.tensor([[-1,-1]]), torch.tensor([[1,1]])
    inputs = [lbs, ubs, None, None]
    net_cnn = CNN(pytorch_model, is_cuda=False)
    vzono_set = net_cnn.reach_over_appr(inputs, test=True)

    vertices = generate_vertices(vzono_set)

    lbs0 = cp.deepcopy(lbs)
    ubs0 = cp.deepcopy(ubs)
    lbs1 = cp.deepcopy(lbs)
    ubs1 = cp.deepcopy(ubs)
    dim_lb = lbs[0, 0]
    dim_ub = ubs[0, 0]

    split_point = (1/2) * (dim_ub - dim_lb) + dim_lb
    ubs0[0, 0] = split_point
    lbs1[0, 0] = split_point
    subset0 = [lbs0, ubs0, None, None]
    subset1 = [lbs1, ubs1, None, None]

    vzono_set0 = net_cnn.reach_over_appr(subset0, test=True)
    vertices0 = generate_vertices(vzono_set0)

    vzono_set1 = net_cnn.reach_over_appr(subset1, test=True)
    vertices1 = generate_vertices(vzono_set1)

    # with open('vertices.pkl', 'wb') as f:
    #     pickle.dump([vertices, vertices0, vertices1], f)

    scipy.io.savemat('vertices.mat',{'vertices': vertices, 'vertices0': vertices0,'vertices1': vertices1})



