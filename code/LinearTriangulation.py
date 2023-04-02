import numpy as np
from utils.MathUtils import getProjectionMatrix


def linearTriangulation(K, C1, R1, C2, R2, x1, x2):
    P1 = getProjectionMatrix(R1, C1, K)
    p1_T, p2_T, p3_T = P1[0], P1[1], P1[2]
    P2 = getProjectionMatrix(R2, C2, K)
    p1p_T, p2p_T, p3p_T = P2[0], P2[1], P2[2]
    X_lst = []
    for i in range(len(x1)):
        x, y = x1[i, 0], x1[i, 1]
        x_p, y_p = x2[i, 0], x2[i, 1]
        A = []
        A.append(y*p3_T - p2_T)
        A.append(p1_T - x*p3_T)
        A.append(y_p*p3p_T - p2p_T)
        A.append(p1p_T - x_p*p3p_T)
        A = np.array(A)
        U, S, V_T = np.linalg.svd(A)
        X = V_T.T[:, -1]
        X = X/X[-1]
        X_lst.append(X)
    return np.array(X_lst)
