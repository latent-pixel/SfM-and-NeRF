import numpy as np


def computeEssentialMatrix(K, F):
    u, s, v_T = np.linalg.svd(F)
    s = np.diag(s)
    # print("Singular values of F:\n", s)
    E = np.dot(K.T, np.dot(F, K))
    U, S, Vh = np.linalg.svd(E)
    S = np.array([[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 0]])
    E = np.dot(U, np.dot(S, Vh))
    # print("Essential Matrix: \n", E)
    return E