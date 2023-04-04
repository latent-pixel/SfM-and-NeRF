import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from utils.MathUtils import eulerAnglesFromRotationMatrix
from matplotlib.path import Path
from matplotlib import transforms
from matplotlib.markers import MarkerStyle


class UnsizedMarker(MarkerStyle):
    def _set_custom_marker(self, path):
        self._transform = transforms.IdentityTransform()
        self._path = path


def createCameraMarker(rotation_y):
    verts = [
    (0., 0.),  # left, bottom
    (0., 0.6),  # left, top
    (0.35, 0.6), # left, cam_bottom
    (0., 1.), # left, cam_top
    (1., 1.),  # right, cam_right
    (0.65, 0.6),  # right, cam_bottom
    (1., 0.6), # right, top
    (1., 0.), #right
    (0., 0.),  # back to left, bottom
    ]
    codes = [
        Path.MOVETO, #begin drawing
        Path.LINETO, #straight line
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.CLOSEPOLY, #close shape. This is not required for this shape but is "good form"
    ]
    cam1 = Path(verts, codes)
    R = transforms.Affine2D().rotate_deg(rotation_y)
    cam2 = cam1.transformed(R)
    # cam_marker = UnsizedMarker(cam2)
    return cam2


def importImages(images_path):
    all_images = []
    for file in sorted(os.listdir(images_path)):
        if ".png" in file:
            image = cv2.imread(os.path.join(images_path,file))
            all_images.append(image)
    return all_images


def drawMatches(img1, img2, matches, inlier_flags):
    w, h, c = img1.shape
    joined_image = np.concatenate((img1, img2), axis=1)
    img1_pts = matches[:, 3:5].astype(int)
    img2_pts = matches[:, 5:7].astype(int)
    for i in range(img1_pts.shape[0]):
        pt_img1 = (img1_pts[i, 0], img1_pts[i, 1])
        pt_img2 = (h+img2_pts[i, 0], img2_pts[i, 1])
        if inlier_flags[i] == 0:
            joined_image = cv2.circle(joined_image, pt_img1, radius=0, color=(0, 0, 255), thickness=3)
            joined_image = cv2.circle(joined_image, pt_img2, radius=0, color=(0, 0, 255), thickness=3)
            joined_image = cv2.line(joined_image, pt_img1, pt_img2, color=(0, 0, 255), thickness=1)
        if inlier_flags[i] == 1:
            joined_image = cv2.circle(joined_image, pt_img1, radius=0, color=(0, 255, 0), thickness=3)
            joined_image = cv2.circle(joined_image, pt_img2, radius=0, color=(0, 255, 0), thickness=3)
            joined_image = cv2.line(joined_image, pt_img1, pt_img2, color=(0, 255, 0), thickness=1)
    cv2.imshow("matches", joined_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def removeOutliers(X):
    dist = np.linalg.norm(X[:, :3], axis=1)
    mean =np.mean(dist, axis=0)
    sd = np.std(dist, axis=0)
    inlier_idxs = np.where(dist <= mean + sd)
    return X[inlier_idxs]


def plotAllX(X_set):
    plt.figure(figsize=(10, 10))
    colors = ('red', 'cyan', 'greenyellow', 'magenta')
    for idx, X in enumerate(X_set):
        X = removeOutliers(X)
        x, z = X[:, 0], X[:, 2]
        plt.scatter(x, z, marker='.', linewidths=0.5, color=colors[idx])
    plt.show()


def plotX2D(C_set, R_set, X):
    X = removeOutliers(X)
    x, z = X[:, 0], X[:, 2]
    plt.figure(figsize=(10, 10))
    plt.scatter(x, z, marker='.', linewidths=0.5, color='greenyellow')
    for i in range(len(C_set)):
        R = eulerAnglesFromRotationMatrix(R_set[i])
        r_y = R[1]
        if r_y < 0: # check / verify
            r_y = 180 + r_y 
        plt.plot(C_set[i][0], C_set[i][2], marker=createCameraMarker(r_y), markersize=20)
    plt.show()


def plotNLT(C_set, R_set, X, X3D_nl):
    X = removeOutliers(X)
    X3D_nl = removeOutliers(X3D_nl)
    x, z = X[:, 0], X[:, 2]
    x_nl, z_nl = X3D_nl[:, 0], X3D_nl[:, 2]
    plt.figure(figsize=(10, 10))
    plt.scatter(x, z, marker='.', s=3, color='r')
    plt.scatter(x_nl, z_nl, marker='.', s=3, color='b')
    for i in range(len(C_set)):
        R = eulerAnglesFromRotationMatrix(R_set[i])
        r_y = R[1]
        if r_y < 0: # check / verify
            r_y = 180 + r_y 
        plt.plot(C_set[i][0], C_set[i][2], marker=createCameraMarker(r_y), markersize=20)
    plt.show()