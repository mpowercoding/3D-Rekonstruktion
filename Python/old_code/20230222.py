import cv2
import numpy as np
from plyfile import PlyData, PlyElement

# Schritt 1: Laden der Bilder und Extraktion von Merkmalen
img1 = cv2.imread('./Milch/IMG_0682.JPG')
img2 = cv2.imread('./Milch/IMG_0677.JPG')
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

# Schritt 2: Merkmalsmatching und Schätzen der Fundamentalmatrix
matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches = matcher.match(des1, des2)
pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS)

# Schritt 3: Schätzen der Kameramatrix
# K = np.array([[1.15422732e+03, 0.00000000e+00, 6.71627794e+02],
#               [0.00000000e+00, 1.14818221e+03, 3.86046312e+02],
#               [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
K = np.load("../K.npy")


E, _ = cv2.findEssentialMat(pts1, pts2, K)
points, R, t, mask = cv2.recoverPose(E, pts1, pts2, K)

# Schritt 4: Rekonstruktion der 3D-Punkte
P1 = np.hstack((np.eye(3), np.zeros((3, 1))))
P2 = np.hstack((R, t))
points4D = cv2.triangulatePoints(P1, P2, pts1, pts2)
points3D = points4D / points4D[3]
points3D = points3D[:3].T

# Schritt 5: Speichern der 3D-Punkte in einer PLY-Datei
vertex = np.core.records.fromarrays(points3D.transpose(), names='x, y, z', formats='f4, f4, f4')
face = np.zeros((0,), dtype=[('vertex_indices', 'i4', (3,))])
plydata = PlyData([PlyElement.describe(vertex, 'vertex'), PlyElement.describe(face, 'face')])
plydata.write('output.ply')
