import numpy as np
import matplotlib.pyplot as plt
from utils.ExtractMatchesFromText import createMatchesArray
from utils.PlotUtils import *
from EstimateFundamentalMatrix import estimateFundamentalMatrix
from GetInlierRANSAC import getInliers
from EssentialMatrixFromFundamentalMatrix import computeEssentialMatrix
from ExtractCameraPose import extractCameraPose
from LinearTriangulation import linearTriangulation
from DisambiguateCameraPose import disambiguateCameraPose
from NonlinearTriangulation import NonlinearTriangulation


data_dir = 'P3Data'
all_images = importImages(data_dir)

K = np.array([[531.122155322710, 0, 407.192550839899],
    [0, 531.541737503901, 313.308715048366],
    [0, 0, 1]])

matches = createMatchesArray(data_dir)
img1, img2 = all_images[0], all_images[1]

inliers = np.empty_like(matches)
F_inliers = np.empty_like(matches)
is_inlier = np.empty_like(matches)
for i in range(len(matches)-1):
    for j in range(i+1, len(matches)):
        if i == 0 and j == 1:
            print("Processing matches for images {} and {}".format(i+1, j+1))
            matches_subset = np.array(matches[i, j])
            inlier_idxs, F_inliers[i, j] = getInliers(matches_subset)
            inliers[i, j] = matches_subset[inlier_idxs]
            is_inlier[i, j] = np.zeros(shape=(len(matches_subset), ))
            is_inlier[i, j][inlier_idxs] = 1
print("Flag 1: Inliers found")

# For the first two images:
matches12 = inliers[0, 1]
# drawMatches(img1, img2, np.array(matches[0, 1]), is_inlier[0, 1])

F = estimateFundamentalMatrix(matches12)
print("Fundamental Matrix: \n", F)
E = computeEssentialMatrix(K, F)
camera_poses = extractCameraPose(E)

C_set = [np.zeros((3, 1))]
R_set = [np.eye(3)]
X_setCR = [] 
for (C, R) in camera_poses:
    X_lst = linearTriangulation(K, C_set[0], R_set[0], C, R, matches12[:, 3:5], matches12[:, 5:7])
    X_setCR.append(X_lst)

# plotAllX(X_setCR)

# C_set = [np.zeros((3, 1)), camera_poses[0][0]]
# R_set = [np.eye(3), camera_poses[0][1]]
# plotX2D(C_set, R_set, X_setCR[0])

# C_set = [np.zeros((3, 1)), camera_poses[1][0]]
# R_set = [np.eye(3), camera_poses[1][1]]
# plotX2D(C_set, R_set, X_setCR[1])

# C_set = [np.zeros((3, 1)), camera_poses[2][0]]
# R_set = [np.eye(3), camera_poses[2][1]]
# plotX2D(C_set, R_set, X_setCR[2])

# C_set = [np.zeros((3, 1)), camera_poses[3][0]]
# R_set = [np.eye(3), camera_poses[3][1]]
# plotX2D(C_set, R_set, X_setCR[3])

C_best, R_best, X_set = disambiguateCameraPose(camera_poses, X_setCR)
C_set.append(C_best)
R_set.append(R_best)
# plotX2D(C_set, R_set, X_set)

X_refined = NonlinearTriangulation(X_set, matches12, K, C_set[0], R_set[0], C_set[1], R_set[1]) # Non-linear Triangulation
# plotNLT(C_set, R_set, X_set, X_refined)
plotReprojection(img1, matches12[:, 3:5], X_set, X_refined, C_set[0], R_set[0], K)