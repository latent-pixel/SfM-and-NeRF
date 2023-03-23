import numpy as np
from EstimateFundamentalMatrix import estimateFundamentalMatrix


def getInliers(matches, n_iter = 1000, eps = 0.01):
    inliers_idx = []
    best_F = None
    for i in range(n_iter):
        matches_batch = matches[np.random.choice(len(matches), 8, replace=False)]
        F = estimateFundamentalMatrix(matches_batch)
        s = []
        for j in range(len(matches)):
            x1 = np.array([matches[j][3], matches[j][4], 1.])
            x2 = np.array([matches[j][5], matches[j][6], 1.])
            if abs(np.dot(x2.T, np.dot(F, x1))) < eps:
                s.append(j)

        if len(inliers_idx) < len(s):
            inliers_idx = s
            best_F = F

    inliers_set = matches[inliers_idx]
    return inliers_set
