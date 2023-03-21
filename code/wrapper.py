import numpy as np
from utils.ExtractMatchesFromText import createMatchesArray
from EstimateFundamentalMatrix import estimateFundamentalMatrix
from GetInlierRANSAC import getInliers

matches = createMatchesArray('P3Data')
test_pts = np.array(matches[0, 1])
print(test_pts.shape)
# F = estimateFundamentalMatrix(test_pts)
inliers_set = getInliers(test_pts)
print(inliers_set.shape)