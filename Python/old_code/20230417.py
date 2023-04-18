import cv2
import numpy
import numpy as np

img1 = cv2.imread('./aircraft/IMG_0766.JPG')
img2 = cv2.imread('./aircraft/IMG_0767.JPG')
K = np.load("../K.npy")
D = np.load("../dist_coeffs.npy")

# Größe der Bilder ermitteln
height, width = img2.shape[:2]

# Abbildung der Bildpunkte berechnen
mapx, mapy = cv2.initUndistortRectifyMap(K, D, None, None, (width, height), cv2.CV_32FC1)

# Rektifizierte Ausgabebilder erzeugen
img_input_rectified = cv2.remap(img1, mapx, mapy, cv2.INTER_LINEAR)
img_output_rectified = cv2.remap(img2, mapx, mapy, cv2.INTER_LINEAR)

# Rektifizierte Ausgabebilder speichern
both_images = numpy.hstack((img_input_rectified, img_output_rectified))
cv2.imwrite('both_rectified.jpg', both_images)
