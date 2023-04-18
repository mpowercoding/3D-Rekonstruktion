import cv2
import numpy as np


def get_fundamental_matrix(img1, img2):
    points1, points2 = sift(img1, img2)
    fundamental_matrix, mask = cv2.findFundamentalMat(points1, points2, cv2.FM_RANSAC, 1.0, 0.99)
    # TODO check output for validation because to less points can lead to error
    return fundamental_matrix, points1, points2

def get_fundamental_matrix_by_points(pts1, pts2):
    fundamental_matrix, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 1.0, 0.99)
    return fundamental_matrix

def get_essential_matrix(points1, points2, camera_matrix):
    essential_matrix, _ = cv2.findEssentialMat(points1, points2, camera_matrix, cv2.RANSAC, 0.999, 1.0)
    return essential_matrix


def get_pose(E, pts1, pts2, K):
    _, R, t, _ = cv2.recoverPose(E, pts1, pts2, K)
    return R, t


def sift(img1, img2):
    kp1, des1 = img1.get_features()
    kp2, des2 = img2.get_features()

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


def calculate_image_connections(images):
    for i, image in enumerate(images):
        conn = {}
        for index, img2 in enumerate(images):
            if index == i:
                continue

            fundamental_matrix, points1, points2 = get_fundamental_matrix(image, img2)
            essential_matrix = get_essential_matrix(points1, points2, image.get_camera_matrix())
            R, T = get_pose(essential_matrix, points1, points2, image.get_camera_matrix())

            conn[index] = {
                "fundamental_matrix": fundamental_matrix,
                "essential_matrix": essential_matrix,
                "R": R,
                "T": T
            }
        print("Pose estimation", image.get_image_path())
        image.set_connections(conn)
    print("")


def check_connections(images):
    for i, image in enumerate(images):
        for index, img2 in enumerate(images):
            if i == index:
                continue
            a = image.get_connections()[index]["T"]
            b = img2.get_connections()[i]["T"]
            res = b - a
            print(res[0], res[1], res[2])
