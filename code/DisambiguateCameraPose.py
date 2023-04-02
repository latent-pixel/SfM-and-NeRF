import numpy as np


def disambiguateCameraPose(cam_poses, X_set):
    best_config = None
    min_count = 0
    for idx in range(len(cam_poses)):
        C, R = cam_poses[idx]
        X_lst = X_set[idx]
        r3 = R[2]
        count = 0
        for X in X_lst:
            X_3D = X[:3]
            if np.dot(r3, (X_3D - C)) > 0 and X_3D[2] > 0:
                count += 1
        if count > min_count:
            min_count = count
            best_config = (C, R, X_lst)
    return best_config
