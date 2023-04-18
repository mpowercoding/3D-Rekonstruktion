import cv2
import numpy

import Fundamental


def rectify_stereo_images(K_left, dist_left, K_right, dist_right, R, T):
    R_left, R_right, P_left, P_right, Q, _, _ = cv2.stereoRectify(
        K_left,
        dist_left,
        K_right,
        dist_right,
        (1280, 960),
        R,
        T,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0
    )

    return R_left, R_right, P_left, P_right, Q


def rectify_images(E, pts1, pts2, K):
    R, T = Fundamental.get_pose(E, pts1, pts2, K)
    return R, T


def undistort(K_left, dist_left, R_left, P_left, imgL, K_right, dist_right, R_right, P_right, imgR):
    mapLx, mapLy = cv2.initUndistortRectifyMap(K_left, dist_left, R_left, P_left, imgL.shape[:2], cv2.CV_32FC1)
    mapRx, mapRy = cv2.initUndistortRectifyMap(K_right, dist_right, R_right, P_right, imgR.shape[:2], cv2.CV_32FC1)
    rectified_left = cv2.remap(imgL, mapLx, mapLy, cv2.INTER_LINEAR)
    rectified_right = cv2.remap(imgR, mapRx, mapRy, cv2.INTER_LINEAR)
    return rectified_left, rectified_right


def show(rectified_left, rectified_right):
    both_images = numpy.hstack((rectified_left, rectified_right))
    cv2.imshow('Rectified Left and Right Images', both_images)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # img1 = cv2.imread('./out/calib/IMG_0675.JPG', cv2.IMREAD_GRAYSCALE)
    # img2 = cv2.imread('./out/calib/IMG_0677.JPG', cv2.IMREAD_GRAYSCALE)
    img1 = cv2.imread('./Milch/IMG_0675.JPG', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('./Milch/IMG_0677.JPG', cv2.IMREAD_GRAYSCALE)

    camera_matrix = numpy.load("../K.npy")
    dist_coeffs = numpy.load("../dist_coeffs.npy")
    F, pts1, pts2 = Fundamental.get_fundamental_matrix(img1, img2)
    E = Fundamental.get_essential_matrix(pts1, pts2, camera_matrix)
    # rms, K_left_new, dist_left_new, K_right_new, dist_right_new, R, T, E, F = stereocalibrate(camera_matrix, camera_matrix, pts1, pts2, F, dist_coeffs, dist_coeffs)

    R, T = rectify_images(E, pts1, pts2, camera_matrix)
    R_left, R_right, P_left, P_right, Q = rectify_stereo_images(camera_matrix, dist_coeffs, camera_matrix, dist_coeffs,
                                                                R, T)
    rectified_left, rectified_right = undistort(camera_matrix, dist_coeffs, R_left, P_left, img1, camera_matrix,
                                                dist_coeffs, R_right, P_right, img2)

    rectified_left = cv2.resize(rectified_left, (rectified_left.shape[1] // 4, rectified_left.shape[0] // 4))
    rectified_right = cv2.resize(rectified_right, (rectified_right.shape[1] // 4, rectified_right.shape[0] // 4))

    show(rectified_left, rectified_right)
