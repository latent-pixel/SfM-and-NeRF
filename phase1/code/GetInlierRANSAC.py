import numpy as np
from EstimateFundamentalMatrix import estimateFundamentalMatrix


def getInliers(matches, n_iter = 1000, eps = 0.01):
    inliers_idx = []
    best_F = None
    for i in range(n_iter):
        matches_batch = matches[np.random.choice(len(matches), 8)]
        F_mtrx = estimateFundamentalMatrix(matches_batch)
        s = []
        for j in range(len(matches)):
            x1 = np.array([matches[j, 3], matches[j, 4], 1.])
            x2 = np.array([matches[j, 5], matches[j, 6], 1.])
            if np.abs(np.dot(x2.T, np.dot(F_mtrx, x1))) < eps:
                s.append(j)

        if len(inliers_idx) < len(s):
            inliers_idx = s
            best_F = F_mtrx
    return inliers_idx, best_F
