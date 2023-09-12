# Detect if the sequence has duplicated elements.
# 1, 2, 3, 5, 1 --> true
# 4, 2, 3, 5, 1 --> false
# 4, 2, 3, 2, 1 --> true

import numpy as np


def main():
    N = 5000
    max_length = 20
    # mask = np.random.randint(low=7, high=max_length, size=(N))
    # mask = max_length*np.ones(N).astype(np.integer)
    X = np.random.randint(1, 200, size=[max_length, N])
    y = np.ones(N)
    for i in range(N):
        # X[mask[i]:, i] = 0
        # seq = list(X[:mask[i], i])
        seq = list(X[:, i])
        y[i] = 0 if len(seq) == len(set(seq)) else 1

    with open('X.npy', 'wb') as o_f:
        np.save(o_f, X)
    with open('y.npy', 'wb') as o_f:
        np.save(o_f, y)
    print(f'saved, y.mean={y.mean()}')
    __import__('pdb').set_trace()


if __name__ == "__main__":
    main()
