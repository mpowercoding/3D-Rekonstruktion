import os
import shutil

import cv2
import numpy as np
from pathlib import Path


class Calibration:
    CHECKERBOARD = (6, 9)

    def analyse_images(self, path):
        objpoints = []
        imgpoints = []
        objp = np.zeros((1, self.CHECKERBOARD[0] * self.CHECKERBOARD[1], 3), np.float32)
        objp[0, :, :2] = np.mgrid[0:self.CHECKERBOARD[0], 0:self.CHECKERBOARD[1]].T.reshape(-1, 2)
        subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
        _img_shape = None
        gray = None

        images = os.listdir(path)
        for fname in images:
            fname = os.path.join(path, fname)
            if not os.path.isfile(fname):
                continue
            img = cv2.imread(fname)
            if _img_shape == None:
                _img_shape = img.shape[:2]
            else:
                assert _img_shape == img.shape[:2], "All images must share the same size."
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, self.CHECKERBOARD,
                                                     cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK)
            # If found, add object points, image points (after refining them)
            if ret == True:
                objpoints.append(objp)
                cv2.cornerSubPix(gray, corners, (3, 3), (-1, -1), subpix_criteria)
                imgpoints.append(corners)

        return objpoints, imgpoints, gray, _img_shape

    def calibrate_camera(self, gray, objpoints, imgpoints, _img_shape):
        DIM = _img_shape[::-1]
        retval, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[:2],
                                                                               None, None)
        # retval, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        np.save("K", camera_matrix)
        np.save("dist_coeffs", dist_coeffs)

        return camera_matrix, DIM, dist_coeffs

    def undistort2(self, img_path, target_dir, camera_matrix, DIM, source_path, dist_coeffs, mode):
        sourcelist = os.listdir(img_path)
        self.check_for_dir(target_dir)
        for image in sourcelist:
            source = os.path.join(source_path, image)
            target = os.path.join(target_dir, image)
            self.undistort_image3(source, target, camera_matrix, dist_coeffs, DIM, mode)

    @staticmethod
    def check_for_dir(path):
        Path(path).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def empty_dir(folder):
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))

    def undistort_image(self, source, target, map1, map2):
        img = cv2.imread(source)
        undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)
        self.save_image(undistorted_img, target)

    def undistort_image2(self, source, target, camera_matrix, dist_coeffs):
        img = cv2.imread(source)
        undistorted_img = cv2.undistort(img, camera_matrix, dist_coeffs, None, camera_matrix)
        self.save_image(undistorted_img, target)

    def undistort_image3(self, source, target, camera_matrix, dist_coeffs, DIM, crop):
        img = cv2.imread(source)
        width = DIM[0]
        height = DIM[1]
        newcameramatrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (width, height), 1, (width, height))
        undistorted_img = cv2.undistort(img, camera_matrix, dist_coeffs, None, newcameramatrix)
        if not crop:
            self.save_image(undistorted_img, target)
        else:
            x, y, w, h = roi
            dst = undistorted_img[y:y + h, x:x + w]
            self.save_image(dst, target)

    @staticmethod
    def save_image(img, path):
        cv2.imwrite(path, img)

    def run(self):
        calib_images_path = 'Testset-SE/Calib/'
        target_path = './out/calib'
        images_path = 'Testset-SE/aircraft/'

        self.empty_dir(target_path)
        objpoints, imgpoints, gray, _img_shape = self.analyse_images(calib_images_path)
        camera_matrix, dimension, dist_coeffs = self.calibrate_camera(gray, objpoints, imgpoints, _img_shape)
        self.undistort2(images_path, target_path, camera_matrix, dimension, images_path, dist_coeffs, False)


if __name__ == "__main__":
    Calibration().run()
