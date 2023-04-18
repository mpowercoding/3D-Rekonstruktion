import numpy as np
import cv2 as cv
import glob

# Load previously saved data
with np.load('B.npz') as X:
    mtx, dist, _, _ = [X[i] for i in ('mtx', 'dist', 'rvecs', 'tvecs')]


class PoseEstimation:
    def run(self):
        pass
