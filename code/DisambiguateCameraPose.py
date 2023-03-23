import numpy as np


def disambiguateCameraPose(cam_poses, X_set):
    best_config = None
    min_count = 0
    for idx, (C, R) in enumerate(cam_poses):
        # print("C: \n", C, "\nand R: \n", R)
        r3 = R[2]
        X_lst = X_set[idx]
        count = 0
        for X in X_lst:
            if np.dot(r3, (X[:-1] - C)) > 0:
                count += 1
        if count > min_count:
            min_count = count
            best_config = (C, R, X_lst)
    return best_config

