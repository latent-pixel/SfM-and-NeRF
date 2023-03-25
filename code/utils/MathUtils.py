import numpy as np
from scipy.spatial.transform import Rotation


def eulerAnglesFromRotationMatrix(R):
    r = Rotation.from_matrix(R)
    r_euler = r.as_euler('xyz', degrees=True)
    return r_euler

