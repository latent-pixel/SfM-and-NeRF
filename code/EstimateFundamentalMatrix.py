import numpy as np


def normalizeCorrespondences(img_pts):
    x = img_pts[:, 0]
    y = img_pts[:, 1]
    centroid_x = np.mean(x)
    centroid_y = np.mean(y)

    x_new = x - centroid_x
    y_new = y - centroid_y
    mean_distance = np.mean(np.sqrt(x_new**2 + y_new**2))
    scale = np.sqrt(2)/mean_distance
    
    transform_mtrx = np.eye(3)
    transform_mtrx[0, 0] = scale
    transform_mtrx[0, 2] = -scale*centroid_x
    transform_mtrx[1, 1] = scale
    transform_mtrx[1, 2] = -scale*centroid_y 
    
    hmg_pts = np.hstack((img_pts, np.ones((len(img_pts), 1))))
    normalized_pts = (transform_mtrx.dot(hmg_pts.T)).T

    return normalized_pts, transform_mtrx

    
def estimateFundamentalMatrix(matches):
    x, T = normalizeCorrespondences(matches[:, 3:5])
    x_p, T_p = normalizeCorrespondences(matches[:, 5:7])
    A_matrix = []
    for i in range(len(matches)):
        u, v = x[i, 0], x[i, 1]
        u_p, v_p = x_p[i, 0], x_p[i, 1]
        A_row = [u*u_p, u_p*v, u_p, u*v_p, v*v_p, v_p, u, v, 1]
        A_matrix.append(A_row)
    A_matrix = np.array(A_matrix)
    [U, S, V_T] = np.linalg.svd(A_matrix)
    F_elems = V_T.T[:, -1]
    F = F_elems.reshape((3, 3))

    u, s, v_T = np.linalg.svd(F)
    s = np.diag(s)
    s[2, 2] = 0
    F = np.dot(u, np.dot(s, v_T))
    
    F = np.dot(T_p.T, np.dot(F, T))
    F = F / F[2, 2]
    return F
