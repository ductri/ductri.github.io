import numpy as np


Q = np.random.rand(10, 5)
K = np.random.rand(10, 5)
X1 = np.zeros((10, 10))
for i in range(10):
    for j in range(10):
        X1[i, j] = ((Q[i, :] - K[j, :])**2).sum()
A = np.diag(Q@Q.T) * np.ones(X1.shape)
B = Q@K.T
C = np.diag(K@K.T)
X2 = A.T-2*B+C
