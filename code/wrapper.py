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


data_dir = 'P3Data'
all_images = importImages(data_dir)

K = np.array([[531.122155322710, 0, 407.192550839899],
    [0, 531.541737503901, 313.308715048366],
    [0, 0, 1]])

matches = createMatchesArray(data_dir)
img1, img2 = all_images[0], all_images[1]
# drawMatches(img1, img2, np.array(matches[0, 1]))

inliers = np.empty_like(matches)
F_inliers = np.empty_like(matches)
for i in range(len(matches)-1):
    for j in range(i+1, len(matches)):
        if i == 0 and j == 1:
            print("Processing matches for images {} and {}".format(i+1, j+1))
            matches_subset = np.array(matches[i, j])
            # print(matches_subset)
            inliers[i, j], F_inliers[i, j] = getInliers(matches_subset)
print("Flag 1: Inliers found")

# For the first two images:
matches12 = inliers[0, 1]
# drawMatches(img1, img2, matches12)

F = F_inliers[0, 1]
E = computeEssentialMatrix(K, F)
camera_poses = extractCameraPose(E)

X_setCR = [] 
for (C, R) in camera_poses:
    X_lst = linearTriangulation(K, np.zeros((3, 1)), np.eye(3), C, R, matches12[:, 3:5], matches12[:, 5:7])
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

C_best, R_best, X_best = disambiguateCameraPose(camera_poses, X_setCR)

C_set = [np.zeros((3, 1)), C_best]
R_set = [np.eye(3), R_best]
plotX2D(C_set, R_set, X_best)
