import numpy as np
from scipy.spatial.transform import Rotation


def getProjectionMatrix(R, C, K):
    Rt = np.hstack((R, -np.dot(R, C.reshape(3, 1))))
    P = np.dot(K, Rt)
    return P


def eulerAnglesFromRotationMatrix(R):
    r = Rotation.from_matrix(R)
    r_euler = r.as_euler('xyz', degrees=True)
    return r_euler

