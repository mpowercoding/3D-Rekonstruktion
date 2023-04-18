import cv2
import numpy as np

import PoseEstimation
from V2.Image import Image


def rectify_initundistortrectifymap(img1, img2, K, D):
    # Größe der Bilder ermitteln
    height, width = img2.shape[:2]

    # Abbildung der Bildpunkte berechnen
    mapx, mapy = cv2.initUndistortRectifyMap(K, D, None, None, (width, height), cv2.CV_32FC1)

    # Rektifizierte Ausgabebilder erzeugen
    img1_rectified = cv2.remap(img1, mapx, mapy, cv2.INTER_LINEAR)
    img2_rectified = cv2.remap(img2, mapx, mapy, cv2.INTER_LINEAR)

    return img1_rectified, img2_rectified


def rectify_uncalibrated(img1, img2, pts1, pts2, F):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    _, H1, H2 = cv2.stereoRectifyUncalibrated(pts1, pts2, F, (w1, h1))

    rectified1 = cv2.warpPerspective(img1, H1, (w1, h1))
    rectified2 = cv2.warpPerspective(img2, H2, (w2, h2))

    return rectified1, rectified2


def stereo_rectify(K_left, dist_left, imgL, K_right, dist_right, imgR, R, T):
    size = imgL.shape[:2]
    R_left, R_right, P_left, P_right, Q, _, _ = cv2.stereoRectify(
        K_left,
        dist_left,
        K_right,
        dist_right,
        size,
        R,
        T,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0
    )

    mapLx, mapLy = cv2.initUndistortRectifyMap(K_left, dist_left, R_left, P_left, imgL.shape[:2], cv2.CV_32FC1)
    mapRx, mapRy = cv2.initUndistortRectifyMap(K_right, dist_right, R_right, P_right, imgR.shape[:2], cv2.CV_32FC1)
    rectified_left = cv2.remap(imgL, mapLx, mapLy, cv2.INTER_LINEAR)
    rectified_right = cv2.remap(imgR, mapRx, mapRy, cv2.INTER_LINEAR)
    return rectified_left, rectified_right, Q


if __name__ == "__main__":
    img1 = cv2.imread('../aircraft/IMG_0766.JPG', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('../aircraft/IMG_0767.JPG', cv2.IMREAD_GRAYSCALE)
    K = np.load("../K.npy")
    D = np.load("../dist_coeffs.npy")

    # Rektifizieren mit rectify_initundistortrectifymap
    img1_rectified_map, img2_rectified_map = rectify_initundistortrectifymap(img1, img2, K, D)

    # rectify uncalibrated
    image1 = Image('../aircraft/IMG_0766.JPG', K, D)
    image2 = Image('../aircraft/IMG_0767.JPG', K, D)
    pts1, pts2 = PoseEstimation.sift(image1, image2)
    F = PoseEstimation.get_fundamental_matrix_by_points(pts1, pts2)
    img1_rectified_uncalibrated, img2_rectified_uncalibrated = rectify_uncalibrated(img1, img2, pts1, pts2, F)

    # stereo rectify
    E = PoseEstimation.get_essential_matrix(pts1, pts2, K)
    R, T = PoseEstimation.get_pose(E, pts1, pts2, K)
    img1_stereo_rectified, img2_stereo_rectified, Q = stereo_rectify(K, D, img1, K, D, img2, R, T)

    # Rektifizierte Ausgabebilder speichern
    both_rectify_initundistortrectifymap_images = np.hstack((img1_rectified_map, img2_rectified_map))
    both_rectify_uncalibrated_images = np.hstack((img1_rectified_uncalibrated, img2_rectified_uncalibrated))
    both_stereo_rectified_images = np.hstack((img1_stereo_rectified, img2_stereo_rectified))

    cv2.imwrite('rectify_initundistortrectifymap.jpg', both_rectify_initundistortrectifymap_images)
    cv2.imwrite('rectify_both_rectify_uncalibrated_images.jpg', both_rectify_uncalibrated_images)
    cv2.imwrite('rectify_both_stereo_rectify_images.jpg', both_stereo_rectified_images)
