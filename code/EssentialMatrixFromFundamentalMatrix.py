import numpy as np


def computeEssentialMatrix(K, F):
    E = np.dot(K.T, np.dot(F, K))
    U, S, V_T = np.linalg.svd(E)
    S = np.diag(S)
    S[2, 2] = 0
    E = np.dot(U, np.dot(S, V_T))
    return E