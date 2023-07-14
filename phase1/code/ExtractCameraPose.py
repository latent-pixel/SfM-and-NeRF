import numpy as np


def extractCameraPose(E):
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    U, D, V_T = np.linalg.svd(E)
    
    # finding the four configurations
    C1 = U[:, 2]
    R1 = np.dot(U, np.dot(W, V_T))
    if round(np.linalg.det(R1)) == -1:
        C1 = -C1
        R1 = -R1

    C2 = -U[:, 2]
    R2 = np.dot(U, np.dot(W, V_T))
    if round(np.linalg.det(R2)) == -1:
        C2 = -C2
        R2 = -R2

    C3 = U[:, 2]
    R3 = np.dot(U, np.dot(W.T, V_T))
    if round(np.linalg.det(R3)) == -1:
        C3 = -C3
        R3 = -R3
    
    C4 = -U[:, 2]
    R4 = np.dot(U, np.dot(W.T, V_T))
    if round(np.linalg.det(R4)) == -1:
        C4 = -C4
        R4 = -R4

    return [(C1, R1), (C2, R2), (C3, R3), (C4, R4)]
