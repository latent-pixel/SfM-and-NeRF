import numpy as np
from utils.ExtractMatchesFromText import createMatchesArray
from EstimateFundamentalMatrix import estimateFundamentalMatrix
from GetInlierRANSAC import getInliers
from EssentialMatrixFromFundamentalMatrix import computeEssentialMatrix
from ExtractCameraPose import extractCameraPose
from LinearTriangulation import linearTriangulation
from DisambiguateCameraPose import disambiguateCameraPose


K = np.array([[531.122155322710, 0, 407.192550839899],
    [0, 531.541737503901, 313.308715048366],
    [0, 0, 1]])
matches = createMatchesArray('P3Data')
inliers = np.empty_like(matches)
for i in range(len(matches)-1):
    for j in range(i+1, len(matches)):
        matches_subset = np.array(matches[i, j])
        inliers[i, j] = getInliers(matches_subset)
print("Flag 1: Inliers found")

# For the first two images:
matches12 = inliers[0, 1]
F = estimateFundamentalMatrix(matches12)
E = computeEssentialMatrix(K, F)
camera_poses = extractCameraPose(E)
X_setCR = [] 
for (C, R) in camera_poses:
    X_lst = linearTriangulation(K, np.zeros((3, 1)), np.eye(3), C, R, matches12[:, 3:5], matches12[:, 5:7])
    X_setCR.append(X_lst)

C_best, R_best, X_best = disambiguateCameraPose(camera_poses, X_setCR)
print("The best camera config, C:\n", C_best, "\nR: \n", R_best)
