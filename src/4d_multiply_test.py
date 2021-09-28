import numpy as np


A = np.ones((4, 3, 3, 2))
index = np.array([[0,0,0,0],[1,0,0,0],[2,0,0,0]])
A[index[:,0],index[:,1],index[:,2],index[:,3]] = A[index[:,0],index[:,1],index[:,2],index[:,3]]*2
xx = A[index[:,0],index[:,1],index[:,2],index[:,3]]
yy = 0


