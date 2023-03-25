import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from utils.MathUtils import eulerAnglesFromRotationMatrix


def importImages(images_path):
    all_images = []
    for file in sorted(os.listdir(images_path)):
        if ".png" in file:
            image = cv2.imread(os.path.join(images_path,file))
            all_images.append(image)
    return all_images


def drawMatches(img1, img2, matches):
    w, h, c = img1.shape
    joined_image = np.concatenate((img1, img2), axis=1)
    img1_pts = matches[:, 3:5].astype(int)
    img2_pts = matches[:, 5:7].astype(int)
    for i in range(img1_pts.shape[0]):
        pt_img1 = (img1_pts[i, 0], img1_pts[i, 1])
        pt_img2 = (h+img2_pts[i, 0], img2_pts[i, 1])
        # print(pt_img1, pt_img2)
        joined_image = cv2.circle(joined_image, pt_img1, radius=0, color=(0, 0, 255), thickness=3)
        joined_image = cv2.circle(joined_image, pt_img2, radius=0, color=(0, 0, 255), thickness=3)
        joined_image = cv2.line(joined_image, pt_img1, pt_img2, color=(0, 255, 0), thickness=1)
    cv2.imshow("matches", joined_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    
def plotX2D(C_set, R_set, X):
    x, z = X[:, 0], X[:, 2]
    plt.figure(figsize=(10, 10))
    plt.scatter(x, z, marker='.', linewidths=0.5, color='greenyellow')
    for i in range(len(C_set)):
        R = eulerAnglesFromRotationMatrix(R_set[i])
        r_y = R[1]
        plt.plot(C_set[i][0], C_set[i][2], marker=(3, 0, int(r_y)), markersize=10)
    plt.show()
