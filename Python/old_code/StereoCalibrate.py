import cv2
import numpy

import Fundamental


def stereocalibrate(K_left, K_right, pts1, pts2, F, dist_left, dist_right, img1, img2):
    # Verwende die Schlüsselpunkte als Näherung der Objektpunkte
    objp = numpy.zeros((pts1.shape[0], 3), dtype=numpy.float32)
    objp[:, :2] = numpy.float32([kp.pt for kp in kp1])

    rms, K_left_new, dist_left_new, K_right_new, dist_right_new, R, T, E, F = cv2.stereoCalibrate(
        img1,
        img2,
        K_left,
        dist_left,
        K_right,
        dist_right,
        flags=cv2.CALIB_USE_INTRINSIC_GUESS,
        F=F
    )
    return rms, K_left_new, dist_left_new, K_right_new, dist_right_new, R, T, E, F


if __name__ == "__main__":
    # img1 = cv2.imread('./out/calib/IMG_0682.JPG', cv2.IMREAD_GRAYSCALE)
    # img2 = cv2.imread('./out/calib/IMG_0677.JPG', cv2.IMREAD_GRAYSCALE)
    img1 = cv2.imread('./Milch/IMG_0682.JPG', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('./Milch/IMG_0677.JPG', cv2.IMREAD_GRAYSCALE)

    camera_matrix = numpy.load("../K.npy")
    dist_coeffs = numpy.load("../dist_coeffs.npy")
    F, pts1, pts2 = Fundamental.get_fundamental_matrix(img1, img2)

    stereocalibrate(camera_matrix, camera_matrix, pts1, pts2, F, dist_coeffs, dist_coeffs, img1, img2)
