import numpy as np


def normalizeCorrespondences(img_pts):
    x = img_pts[:, 0]
    y = img_pts[:, 1]
    centroid_x = np.mean(x)
    centroid_y = np.mean(y)

    x_new = x - centroid_x
    y_new = y - centroid_y
    std = np.mean(np.sqrt(x_new**2 + y_new**2))
    scale = np.sqrt(2)/std
    
    transform_mtrx = np.eye(3)
    transform_mtrx[0, 0] = scale
    transform_mtrx[0, 2] = -scale*centroid_x
    transform_mtrx[1, 1] = scale
    transform_mtrx[1, 2] = -scale*centroid_y

    # mean = np.mean(img_pts, axis=0)
    # std = np.std(img_pts)
    # transform_mtrx = np.array([[np.sqrt(2)/std, 0, -np.sqrt(2)/std*mean[0]],
    #               [0, np.sqrt(2)/std, -np.sqrt(2)/std*mean[1]],
    #               [0, 0, 1]])

    hmg_pts = np.hstack((img_pts, np.ones((len(img_pts), 1))))
    normalized_pts = (transform_mtrx.dot(hmg_pts.T)).T

    return normalized_pts[:, :2], transform_mtrx

    
def estimateFundamentalMatrix(matches):
    pts1_norm, T1 = normalizeCorrespondences(matches[:, 3:5])
    pts2_norm, T2 = normalizeCorrespondences(matches[:, 5:7])

    A_matrix = []
    for i in range(len(matches)):
        u1, v1 = pts1_norm[i]
        u2, v2 = pts2_norm[i]
        A_row = [u1*u2, v1*u2, u2, u1*v2, v1*v2, v2, u1, v1, 1]
        A_matrix.append(A_row)
    A_matrix = np.array(A_matrix)
    U, S, Vh = np.linalg.svd(A_matrix)
    F_norm = Vh[-1].reshape(3, 3)

    u, s, vh = np.linalg.svd(F_norm)
    s[-1] = 0
    F_norm = np.dot(u, np.dot(np.diag(s), vh))

    F_ = np.dot(T2.T, np.dot(F_norm, T1))
    F_ = F_ / F_[2, 2]

    return F_
