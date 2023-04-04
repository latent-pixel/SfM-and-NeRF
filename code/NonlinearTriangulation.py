import numpy as np
from scipy.optimize import least_squares
from utils.MathUtils import getProjectionMatrix


# def getReprojectionError(X_3D, matches, K, C1, R1, C2, R2):
#     X_3D = np.array(X_3D)
#     P1 = getProjectionMatrix(R1, C1, K)
#     p1_T, p2_T, p3_T = P1[0], P1[1], P1[2]
#     P2 = getProjectionMatrix(R2, C2, K)
#     p1p_T, p2p_T, p3p_T = P2[0], P2[1], P2[2]
#     total_reproj_error = 0
#     for i in range(matches.shape[0]):
#         u1, v1 = matches[i, 3:5]
#         u2, v2 = matches[i, 5:7]
#         X_tilde = X_3D[i]
#         u1_reproj = np.dot(p1_T, X_tilde) / np.dot(p3_T, X_tilde)
#         v1_reproj = np.dot(p2_T, X_tilde) / np.dot(p3_T, X_tilde)
#         u2_reproj = np.dot(p1p_T, X_tilde) / np.dot(p3p_T, X_tilde)
#         v2_reproj = np.dot(p2p_T, X_tilde) / np.dot(p3p_T, X_tilde)
#         reproj_error = (u1 - u1_reproj)**2 + (v1 - v1_reproj)**2 +\
#                         (u2 - u2_reproj)**2 + (v2 - v2_reproj)**2
#         total_reproj_error += reproj_error
#     return total_reproj_error


def ObjectiveFunction(X_tilde, match_, P1, P2):
    X_tilde = np.array(X_tilde)
    p1_T, p2_T, p3_T = P1[0], P1[1], P1[2]
    p1p_T, p2p_T, p3p_T = P2[0], P2[1], P2[2]
    u1, v1 = match_[3:5]
    u2, v2 = match_[5:7]
    u1_reproj = np.dot(p1_T, X_tilde) / np.dot(p3_T, X_tilde)
    v1_reproj = np.dot(p2_T, X_tilde) / np.dot(p3_T, X_tilde)
    u2_reproj = np.dot(p1p_T, X_tilde) / np.dot(p3p_T, X_tilde)
    v2_reproj = np.dot(p2p_T, X_tilde) / np.dot(p3p_T, X_tilde)
    return (u1 - u1_reproj)**2 + (v1 - v1_reproj)**2 + (u2 - u2_reproj)**2 + (v2 - v2_reproj)**2


def NonlinearTriangulation(X_3D, matches, K, C1, R1, C2, R2):
    P1 = getProjectionMatrix(R1, C1, K)
    P2 = getProjectionMatrix(R2, C2, K)
    X3D_nl = []
    for i in range(matches.shape[0]):
        X0 = X_3D[i]
        match_ = matches[i]
        res = least_squares(ObjectiveFunction, X0, method='trf', ftol=1e-8, args=(match_, P1, P2))  # removed verbose = 2
        X = res.x
        X = X / X[-1]
        X3D_nl.append(X)
    X3D_nl = np.array(X3D_nl)
    return X3D_nl