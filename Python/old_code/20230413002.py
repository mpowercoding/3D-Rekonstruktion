import cv2
import numpy as np
import open3d as o3d

# Lade die Bilder
from V2 import scaling

img1 = cv2.imread('./aircraft/IMG_0766.JPG')
img2 = cv2.imread('./aircraft/IMG_0767.JPG')
K = np.load("../K.npy")
D = np.load("../dist_coeffs.npy")

# Konvertiere die Bilder in Graustufen
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

gray1 = scaling.downsample_image(gray1, 4)
gray2 = scaling.downsample_image(gray2, 4)

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
src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
F, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.FM_RANSAC)
E = np.matmul(np.matmul(np.transpose(K), F), K)
_, R, T, _ = cv2.recoverPose(E, src_pts, dst_pts, K)

# Berechne die epipolare Linien
lines1 = cv2.computeCorrespondEpilines(src_pts, 1, F)
lines2 = cv2.computeCorrespondEpilines(dst_pts, 2, F)

# Rektifiziere die Bilder mit cv2.stereoRectify()
h1, w1 = img1.shape[:2]
h2, w2 = img2.shape[:2]
R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(K, D, K, D, (w1, h1), R, T)
# R_left, R_right, P_left, P_right, Q, _, _

map1x, map1y = cv2.initUndistortRectifyMap(K, D, R1, P1, (w1, h1), cv2.CV_32FC1)
map2x, map2y = cv2.initUndistortRectifyMap(K, D, R2, P2, (w2, h2), cv2.CV_32FC1)
# rectified1 = cv2.remap(img1, map1x, map1y, cv2.INTER_LINEAR)
# rectified2 = cv2.remap(img2, map2x, map2y, cv2.INTER_LINEAR)
rectified1 = cv2.remap(img1, map1x, map1y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
rectified2 = cv2.remap(img2, map2x, map2y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)

# Bestimme die Disparitäten
window_size = 5
min_disp = -1
max_disp = 31
num_disp = max_disp - min_disp

stereo = cv2.StereoSGBM_create(minDisparity=min_disp,
                               numDisparities=num_disp,
                               blockSize=window_size,
                               uniquenessRatio = 5,
                               speckleWindowSize = 5,
                               speckleRange = 2,
                               disp12MaxDiff = 2,
                               P1=8 * 3 * window_size ** 2,
                               P2=32 * 3 * window_size ** 2,
                               # disp12MaxDiff=1,
                               # uniquenessRatio=10,
                               # speckleWindowSize=100,
                               # speckleRange=32
                               )
gray_left = cv2.cvtColor(rectified1, cv2.COLOR_BGR2GRAY)
gray_right = cv2.cvtColor(rectified2, cv2.COLOR_BGR2GRAY)
disparity = stereo.compute(gray_left, gray_right)#.astype(np.float32) / 16.0

# Wandle die Disparitäten in 3D-Punkte um
disparity = np.float32(np.divide(disparity, 16.0))
points_3d = cv2.reprojectImageTo3D(disparity, Q, handleMissingValues=False)
colors = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
mask_map = disparity > disparity.min()

outputpoints = points_3d[mask_map]
outputcolors = colors[mask_map]

def create_point_cloud_file(vertices, colors, filename):
    colors = colors.reshape(-1,3)
    vertices = vertices.reshape(-1,3)
    vertices = np.hstack([vertices, colors])

    ply_header = '''ply
    format ascii 1.0
    element vertex %(vert_num)d
    property float x
    property float y
    property float z
    property uchar red
    property uchar green
    property uchar blue
    ent_header
    '''
    with open(filename, 'w') as f:
        f.write(ply_header %dict(vert_num=len(vertices)))
        np.savetxt(f, vertices, '%f %f %f %d %d %d')

outputfile = "idk.ply"
create_point_cloud_file(outputpoints, outputcolors, outputfile)

# Speichere die Ergebnisse
def show(rectified_left, rectified_right):
    both_images = np.hstack((rectified_left, rectified_right))
    cv2.imwrite('./out/test/rectified3.png', both_images)
    # cv2.imshow('Rectified Left and Right Images', both_images)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

cv2.imwrite('./out/test/rectified1.png', rectified1)
cv2.imwrite('./out/test/rectified2.png', rectified2)
cv2.imwrite('./out/test/disparity.png', (disparity - min_disp) / num_disp * 255)
cv2.imwrite('./out/test/points_3d.png', points_3d)
show(rectified1, rectified2)

# Konvertiere die 3D-Punkte in eine open3d-Punktwolke
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(outputpoints.reshape(-1, 3))

# Exportiere die Punktwolke als OBJ-Dateix
o3d.io.write_point_cloud("point_cloud3.ply", point_cloud)
