import numpy as np


# A = np.ones((4, 3, 3, 2))
# index = np.array([[0,0,0,0],[1,0,0,0],[2,0,0,0]])
# A[index[:,0],index[:,1],index[:,2],index[:,3]] = A[index[:,0],index[:,1],index[:,2],index[:,3]]*2
# xx = A[index[:,0],index[:,1],index[:,2],index[:,3]]
# yy = 0


import torch

A = torch.ones((2,3,3,3))
v = torch.tensor([[[[1]],[[2]],[[3]]]])
v2 = torch.tensor([1,2,3])
v3 = torch.reshape(v2, (1,len(v2),1,1))
xx = A - v3
yy = 0

