import cv2
import numpy as np
import os
from plyfile import PlyData, PlyElement

# Schritt 1: Laden der Bilder und Extraktion von Merkmalen
image_dir = '../Testset-SE/Milch/'
images = sorted(os.listdir(image_dir))
sift = cv2.SIFT_create()
des_list = []
kp_list = []
for i, fname in enumerate(images):
    img = cv2.imread(os.path.join(image_dir, fname))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kp, des = sift.detectAndCompute(gray, None)
    kp_list.append(kp)
    des_list.append(des)

# Schritt 2: Merkmalsmatching und Schätzen der Fundamentalmatrix
matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
F_list = []
for i in range(len(images) - 1):
    matches = matcher.match(des_list[i], des_list[i+1])
    pts1 = np.float32([kp_list[i][m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp_list[i+1][m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS)
    F_list.append(F)

# Schritt 3: Schätzen der Kameraparameter
# K = np.array([[1.15422732e+03, 0.00000000e+00, 6.71627794e+02],
#               [0.00000000e+00, 1.14818221e+03, 3.86046312e+02],
#               [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
K = np.load("../K.npy")

E_list = []
R_list = []
t_list = []
for i in range(len(images)):
    if i == 0:
        E_list.append(np.eye(3))
        R_list.append(np.eye(3))
        t_list.append(np.zeros((3, 1)))
    else:
        E, mask = cv2.findEssentialMat(pts1, pts2, K)
        points, R, t, mask = cv2.recoverPose(E, pts1, pts2, K)
        E_list.append(E)
        R_list.append(R)
        t_list.append(t)
    pts1 = pts2

# Schritt 4: Triangulation der 3D-Punkte
P_list = []
point_3d_list = []
depth_list = []
for i in range(len(images)):
    P = np.hstack((np.eye(3), np.zeros((3, 1))))
    if i == 0:
        P_list.append(P)
    else:
        E = np.cross(t_list[i], R_list[i])
        P = np.hstack((R_list[i], -np.dot(R_list[i], t_list[i])))
        P_list.append(P)
        if i < len(images) - 1:
            F = F_list[i-1]
            Kinv = np.linalg.inv(K)
            epipole = np.dot(Kinv, np.array([1, 1, 1]))
            epipole = epipole / epipole[2]
            line = np.dot(F, epipole)
            line = line / np.sqrt(line[0]**2 + line[1]**2)
            pts, _, _ = cv2.decomposeProjectionMatrix(P_list[i])
            pts = np.array([pt[:3] / pt[3] for pt in pts]).T
            proj_point = cv2.triangulatePoints(P_list[i], P_list[i+1], pts1, pts2)
            proj_point = proj_point / proj_point[3]
            depth = proj_point[2]
            point_3d = cv2.convertPointsFromHomogeneous(proj_point.T)
            point_3d = point_3d.reshape(-1, 3)
            point_3d_list.append(point_3d)
            depth_list.append(depth)

# Schritt 5: Schreiben der 3D-Punkte in eine PLY-Datei
vertices = np.concatenate(point_3d_list, axis=0)
vertices = vertices.astype([('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
faces = np.zeros((0, 3), dtype=np.int32)
for i in range(len(point_3d_list) - 1):
    num_points = point_3d_list[i].shape[0]
    faces = np.vstack((faces, np.column_stack((np.arange(num_points), np.arange(num_points) + num_points, np.arange(num_points) + 1))))
    faces = np.vstack((faces, np.column_stack((np.arange(num_points) + 1, np.arange(num_points) + num_points, np.arange(num_points) + num_points + 1))))
vertices = np.array([tuple(vertex) for vertex in vertices])
faces = np.array([tuple(face) for face in faces])
plydata = PlyData([PlyElement.describe(vertices, 'vertex'), PlyElement.describe(faces, 'face')])
plydata.write('output.ply')
