import cv2
import numpy as np
from plyfile import PlyData, PlyElement

img_left = cv2.imread('./Milch/IMG_0682.JPG')
img_right = cv2.imread('./Milch/IMG_0677.JPG')


# Erstellen Sie den Feature-Detektor und Deskriptor-Extraktor.
detector = cv2.SIFT_create()
matcher = cv2.BFMatcher()

# Berechnen Sie die Schlüsselpunkte und Descriptoren.
kp1, des1 = detector.detectAndCompute(img_left, None)
kp2, des2 = detector.detectAndCompute(img_right, None)

# Finden Sie Korrespondenzen zwischen den Descriptoren.
matches = matcher.knnMatch(des1, des2, k=2)



# Definieren Sie die minimale Anzahl von Übereinstimmungen, die erforderlich sind, um eine gute Übereinstimmung zu sein.
MIN_MATCH_COUNT = 10

good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

if len(good_matches) > MIN_MATCH_COUNT:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
else:
    print("Not enough matches are found - %d/%d" % (len(good_matches), MIN_MATCH_COUNT))



# Schätzen Sie die fundamentale Matrix.
F, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.FM_RANSAC)

# Schätzen Sie die essentielle Matrix.
E, _ = cv2.findEssentialMat(src_pts, dst_pts, focal=1.0, pp=(0, 0), method=cv2.RANSAC, prob=0.999, threshold=3.0)


# Definieren Sie die interne Kameramatrix.
#np.save('calib', [mtx,dist,rvecs,tvecs])
# K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
K = np.load("../K.npy")

# Schätzen Sie die externe Kameramatrix für das linke Bild.
R1, t1 = None, None
_, R1, t1, _ = cv2.recoverPose(E, src_pts, dst_pts, cameraMatrix=K)

# Schätzen Sie die externe Kameram

# Schätzen Sie die externe Kameramatrix für das rechte Bild.
R2, t2 = None, None
_, R2, t2, _ = cv2.recoverPose(E, dst_pts, src_pts, cameraMatrix=K)

# Konstruieren Sie die Projektionsmatrizen für die beiden Bilder.
P1 = np.hstack((K, np.zeros((3, 1))))
P2 = np.hstack((np.dot(K, np.hstack((R2, t2))), t2.reshape(3, 1)))

# Berechnen Sie die Disparitätskarte und erstellen Sie die Tiefenkarte.
stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
disparity = stereo.compute(gray_left, gray_right)
depth = np.divide(float(K[0, 0]), disparity)

# Zeigen Sie die Tiefenkarte an.
depth = cv2.resize(depth, (800, 1067))
cv2.imshow('Depth Map', depth)
cv2.waitKey(10000)
cv2.destroyAllWindows()


# Annahme: die Kamera- und Kalibrierungsparameter wurden bereits definiert.

# K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
# Konvertieren Sie die Tiefenkarte in 3D-Punktwolke.
h, w = depth.shape
X, Y = np.meshgrid(np.arange(w), np.arange(h))
Z = depth

cx = K[0, 2]
cy = K[1, 2]
fx = K[0, 0]
fy = K[1, 1]

X = (X - cx) * Z / fx
Y = (Y - cy) * Z / fy
points = np.dstack((X, Y, Z)).reshape(-1, 3)

# Erstellen Sie eine PlyData-Instanz und fügen Sie die Punkte hinzu.
vertex = np.core.records.fromarrays(points.transpose(), names='x, y, z', formats='f4, f4, f4')
face = np.zeros((0,), dtype=[('vertex_indices', 'i4', (3,))])
plydata = PlyData([PlyElement.describe(vertex, 'vertex'), PlyElement.describe(face, 'face')])

# Schreiben Sie die PlyData-Instanz in eine PLY-Datei.
plydata.write('depth_map.ply')
