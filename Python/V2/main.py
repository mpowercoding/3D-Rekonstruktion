import os

import numpy as np
from plyfile import PlyData, PlyElement

import PoseEstimation
from V2 import Reconstruction, Rectify
from V2.Image import Image


def images_to_list(path_list, path, camera_matrix, dist_coeffs):
    images = list()
    for index, filename in enumerate(path_list):
        print(index, filename)
        images.append(Image(os.path.join(path, filename), camera_matrix, dist_coeffs))
    print("")
    return images


# if __name__ == "__main__":
#     camera_matrix = np.load("../K.npy")
#     dist_coeffs = np.load("../dist_coeffs.npy")
#     path = '../Testset-power/bank/'
#     target_path = '../out/depth'
#
#     images_paths = os.listdir(path)
#     images = images_to_list(images_paths, path, camera_matrix, dist_coeffs)
#
#     PoseEstimation.calculate_image_connections(images)
#     PoseEstimation.check_connections(images)

if __name__ == "__main__":
    K = np.load("../K.npy")
    D = np.load("../dist_coeffs.npy")
    image1 = Image('../aircraft/IMG_0766.JPG', K, D)
    image2 = Image('../aircraft/IMG_0767.JPG', K, D)
    pts1, pts2 = PoseEstimation.sift(image1, image2)
    F = PoseEstimation.get_fundamental_matrix_by_points(pts1, pts2)
    E = PoseEstimation.get_essential_matrix(pts1, pts2, K)
    R, T = PoseEstimation.get_pose(E, pts1, pts2, K)
    img1_stereo_rectified, img2_stereo_rectified, Q = Rectify.stereo_rectify(K, D, image1.get_image(),
                                                                             K, D, image2.get_image(), R, T)

    focal_length1 = Image.get_focal_length('../Testset-SE/aircraft/IMG_0766.JPG')
    focal_length2 = Image.get_focal_length('../Testset-SE/aircraft/IMG_0767.JPG')
    points_3D = Reconstruction.reconstruct2(img1_stereo_rectified, img2_stereo_rectified, Q)

    # Schreiben der 3D-Punkte als PLY-Datei
    h, w = img1_stereo_rectified.shape[:2]
    vertices = np.zeros(h*w, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    vertices['x'] = points_3D[:,:,0].ravel()
    vertices['y'] = points_3D[:,:,1].ravel()
    vertices['z'] = points_3D[:,:,2].ravel()
    ply = PlyData([PlyElement.describe(vertices, 'vertex')], text=True)
    ply.write('points.ply')
