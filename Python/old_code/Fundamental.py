import cv2
import numpy
import numpy as np


def get_fundamental_matrix(img1, img2):
    points1, points2 = sift(img1, img2)
    fundamental_matrix, mask = cv2.findFundamentalMat(points1, points2, cv2.FM_RANSAC, 1.0, 0.99)

    # TODO check output for validation because to less points can lead to error
    return fundamental_matrix, points1, points2


def get_essential_matrix(points1, points2, camera_matrix):
    essential_matrix, _ = cv2.findEssentialMat(points1, points2, camera_matrix, cv2.RANSAC, 0.999, 1.0)
    return essential_matrix

def get_pose(E, pts1, pts2, K):
    _, R, t, _ = cv2.recoverPose(E, pts1, pts2, K)
    return R, t

def sift(img1, img2):
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    # FLANN matching
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    pts1 = []
    pts2 = []

    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.8 * n.distance:
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)

    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)

    return pts1, pts2


if __name__ == "__main__":
    # img1 = cv2.imread('./out/calib/IMG_0682.JPG', cv2.IMREAD_GRAYSCALE)
    # img2 = cv2.imread('./out/calib/IMG_0677.JPG', cv2.IMREAD_GRAYSCALE)
    img1 = cv2.imread('./Milch/IMG_0682.JPG', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('./Milch/IMG_0677.JPG', cv2.IMREAD_GRAYSCALE)

    camera_matrix = np.load("../K.npy")

    F, pts1, pts2 = get_fundamental_matrix(img1, img2)
    E = get_essential_matrix(pts1, pts2, camera_matrix)

    numpy.save("F", F)
    numpy.save("E", E)
