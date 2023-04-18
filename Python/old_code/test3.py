import os

import cv2
import numpy as np

CHECKERBOARD = (6, 9)
subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_CHECK_COND + cv2.fisheye.CALIB_FIX_SKEW
objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
_img_shape = None
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.
images = os.listdir('../Testset-11/Calib/')
gray = None
for fname in images:
    fname = os.path.join("../Testset-11/Calib", fname)
    if not os.path.isfile(fname):
        continue
    img = cv2.imread(fname)
    if _img_shape == None:
        _img_shape = img.shape[:2]
    else:
        assert _img_shape == img.shape[:2], "All images must share the same size."
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        cv2.cornerSubPix(gray, corners, (3, 3), (-1, -1), subpix_criteria)
        imgpoints.append(corners)

DIM = _img_shape[::-1]
K = np.zeros((3, 3))
D = np.zeros((4, 1))
rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(6)]
tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(9)]
rms, _, _, _, _ = \
    cv2.fisheye.calibrate(
        objpoints,
        imgpoints,
        gray.shape[::-1],
        K,
        D,
        rvecs,
        tvecs,
        calibration_flags,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
    )

map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)

img = cv2.imread('./Calib/IMG_2592.jpg')
undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)

x, y = undistorted_img.shape[1], undistorted_img.shape[0]
undistorted_img = cv2.resize(undistorted_img, (round(x * 0.2), round(y * 0.2)))

cv2.imshow('img', undistorted_img)
cv2.waitKey(10000)

# image = cv2.imread('./Calib/IMG_2592.jpg')
#
# distCoeff = np.zeros((4, 1), np.float64)
# distCoeff[0, 0] = 6
# distCoeff[1, 0] = 9
# distCoeff[2, 0] = 0
# distCoeff[3, 0] = 0
#
# cam = np.eye(3, dtype=np.float32)
# cam[0, 2] = image.shape[1] / 2.0  # define center x
# cam[1, 2] = image.shape[0] / 2.0  # define center y
# cam[0, 0] = 10.  # define focal length x
# cam[1, 1] = 10.  # define focal length y
# nk = cam.copy()
# scale = 2
# nk[0, 0] = cam[0, 0] / scale  # scaling
# nk[1, 1] = cam[1, 1] / scale
#
# undistorted = cv2.undistort(image, cam, distCoeff, None, nk)
# cv2.destroyAllWindows()
# cv2.imshow('img', undistorted_img)
# cv2.waitKey(5000)
