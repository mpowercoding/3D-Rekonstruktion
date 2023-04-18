import cv2
import numpy
import numpy as np
import open3d as o3d

# Lade die Bilder
img1 = cv2.imread('./aircraft/IMG_0765.JPG')
img2 = cv2.imread('./aircraft/IMG_0767.JPG')

# Konvertiere die Bilder in Graustufen
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Finde die Eckpunkte der markanten Merkmale in den Bildern
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

# Finde die Übereinstimmungen der markanten Merkmale
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

# Filtere die Übereinstimmungen anhand der Lowe's Ratio Test
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

# Berechne die Fundamentalmatrix und die essentielle Matrix
K = numpy.load("../K.npy")
src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
F, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.FM_RANSAC)
E = np.matmul(np.matmul(np.transpose(K), F), K)

# Berechne die epipolare Linien
lines1 = cv2.computeCorrespondEpilines(src_pts, 1, F)
lines2 = cv2.computeCorrespondEpilines(dst_pts, 2, F)

# Rektifiziere die Bilder
h1, w1 = img1.shape[:2]
h2, w2 = img2.shape[:2]
_, H1, H2 = cv2.stereoRectifyUncalibrated(src_pts, dst_pts, F, (w1, h1))

rectified1 = cv2.warpPerspective(img1, H1, (w1, h1))
rectified2 = cv2.warpPerspective(img2, H2, (w2, h2))
#
# # Speichere die rektifizierten Bilder
# cv2.imwrite('bild1_rectified.jpg', rectified1)
# cv2.imwrite('bild2_rectified.jpg', rectified2)
# both_images = np.hstack((rectified1, rectified2))
# cv2.imwrite('rectified3.png', both_images)


def show(rectified_left, rectified_right):
    both_images = numpy.hstack((rectified_left, rectified_right))
    cv2.imshow('Rectified Left and Right Images', both_images)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# show(rectified1, rectified2)


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

disparity = stereo.compute(rectified1, rectified2)

# Normalize the values to a range from 0..255 for a grayscale image
disparity = cv2.normalize(disparity, disparity, alpha=255,
                          beta=0, norm_type=cv2.NORM_MINMAX)
disparity = numpy.uint8(disparity)

# Perspective Transform anwenden
# disparity = cv2.resize(disparity, (disparity.shape[1] // 4, disparity.shape[0] // 4))
# cv2.imshow("Disparity", disparity)
# cv2.waitKey(0)
cv2.imwrite("20230413.png", disparity)

f = K[1, 1] + K[2, 2] / 2
Q = np.float32([[1, 0, 0, 0],
                 [0, -1, 0, 0],
                 [0, 0, f * 0.05, 0],
                 [0, 0, 0, 1]])
points_3d = cv2.reprojectImageTo3D(disparity, Q)

# Konvertiere die 3D-Punkte in eine open3d-Punktwolke
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(points_3d.reshape(-1, 3))

# Exportiere die Punktwolke als OBJ-Datei
o3d.io.write_point_cloud("point_cloud2.ply", point_cloud)
