import numpy as np


def linearTriangulation(K, C1, R1, C2, R2, x1, x2):
    C1, R1, C2, R2 = K.dot(C1), K.dot(R1), K.dot(C2), K.dot(R2)
    P1 = np.hstack((R1, C1.reshape((len(C1), -1))))
    P2 = np.hstack((R2, C2.reshape((len(C1), -1))))
    X_lst = []
    for i in range(len(x1)):
        A = []
        A.append(np.dot(x1[i, 1], P1[2].T) - P1[1].T)
        A.append(P1[0].T - np.dot(x1[i, 0], P1[2].T))
        A.append(np.dot(x2[i, 1], P2[2].T) - P2[1].T)
        A.append(P2[0].T - np.dot(x2[i, 0], P2[2].T))
        A = np.array(A)
        U, S, V_T = np.linalg.svd(A)
        X = V_T.T[:, -1]
        X = X/X[-1]
        X_lst.append(X)
    return np.array(X_lst)