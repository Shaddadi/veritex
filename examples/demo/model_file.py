import torch.nn as nn
import torch.nn.functional as F
from scipy.io import loadmat, savemat
import torch

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(3, 7),
            nn.ReLU(),
            nn.Linear(7, 7),
            nn.ReLU(),
            nn.Linear(7, 7),
            nn.ReLU(),
            nn.Linear(7, 7),
            nn.ReLU(),
            nn.Linear(7, 7),
            nn.ReLU(),
            nn.Linear(7, 7),
            nn.ReLU(),
            nn.Linear(7, 7),
            nn.ReLU(),
            nn.Linear(7, 2),
        )

    def forward(self, x):
        x = self.model(x)
        return x


model = Net().model
nn_path = "nets/NeuralNetwork7_3.mat"
filemat = loadmat(nn_path)
W = filemat['W'][0]
b = filemat['b'][0]
n, m = 0, 0
while n<=7:
    if isinstance(model[m], nn.ReLU):
        m += 1
        continue
    weights = nn.Parameter(torch.tensor(W[n]), requires_grad=False)
    bias = nn.Parameter(torch.tensor(b[n][:,0]), requires_grad=False)
    with torch.no_grad():
        model[m].weight = weights
        model[m].bias = bias
        m += 1
        n += 1


torch.save(model, 'demo_model.pt')
model = torch.load('demo_model.pt')
model.eval()