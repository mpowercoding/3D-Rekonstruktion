import cv2
import numpy

import Fundamental
import Rectification


def process_image_pair(img1, img2, outputname, camera_matrix, dist_coeffs):
    F, pts1, pts2 = Fundamental.get_fundamental_matrix(img1, img2)
    E = Fundamental.get_essential_matrix(pts1, pts2, camera_matrix)
    # rms, K_left_new, dist_left_new, K_right_new, dist_right_new, R, T, E, F = stereocalibrate(camera_matrix, camera_matrix, pts1, pts2, F, dist_coeffs, dist_coeffs)

    R, T = Rectification.extraxt_rotation_and_transformaiton(E, pts1, pts2, camera_matrix)

    image_size = img1.shape[::-1]
    R_left, R_right, P_left, P_right, Q = Rectification.rectify_stereo_images(camera_matrix, dist_coeffs, camera_matrix,
                                                                              dist_coeffs,
                                                                              R, T, image_size)

    rectified_left, rectified_right = Rectification.undistort(camera_matrix, dist_coeffs, R_left, P_left, img1,
                                                              camera_matrix,
                                                              dist_coeffs, R_right, P_right, img2)

    cv2.imwrite(outputname + "_rectified.png", numpy.hstack((rectified_left, rectified_right)))

    # 3d here
    block_size = 11
    min_disp = -128
    max_disp = 128
    num_disp = max_disp - min_disp
    uniquenessRatio = 5
    speckleWindowSize = 200
    speckleRange = 2
    disp12MaxDiff = 0
    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=block_size,
        uniquenessRatio=uniquenessRatio,
        speckleWindowSize=speckleWindowSize,
        speckleRange=speckleRange,
        disp12MaxDiff=disp12MaxDiff,
        P1=8 * 1 * block_size * block_size,
        P2=32 * 1 * block_size * block_size,
    )

    disparity = stereo.compute(rectified_left, rectified_right)

    # Normalize the values to a range from 0..255 for a grayscale image
    disparity = cv2.normalize(disparity, disparity, alpha=255,
                                  beta=0, norm_type=cv2.NORM_MINMAX)
    disparity = numpy.uint8(disparity)
    points_3d = cv2.reprojectImageTo3D(disparity, Q)


    # Perspective Transform anwenden
    # disparity = cv2.resize(disparity, (disparity.shape[1] // 4, disparity.shape[0] // 4))
    # cv2.imshow("Disparity", disparity)
    # cv2.waitKey(0)
    cv2.imwrite(outputname + ".png", points_3d)


if __name__ == "__main__":
    # img1 = cv2.imread('./out/calib/IMG_0675.JPG', cv2.IMREAD_GRAYSCALE)
    # img2 = cv2.imread('./out/calib/IMG_0677.JPG', cv2.IMREAD_GRAYSCALE)
    # img1 = cv2.imread('./Milch/IMG_0681.JPG', cv2.IMREAD_GRAYSCALE)
    # img2 = cv2.imread('./Milch/IMG_0677.JPG', cv2.IMREAD_GRAYSCALE)
    img1 = cv2.imread('./aircraft/IMG_0765.JPG', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('./aircraft/IMG_0766.JPG', cv2.IMREAD_GRAYSCALE)
