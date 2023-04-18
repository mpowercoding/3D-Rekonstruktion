import ctypes
import os
import numpy as np
import cv2


def harrisCornerDetection(path):

    #Bild laden
    image = cv2.imread(path)

    #Konvertieren des Bildes
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)

    #Andwenden der Harris-Corner-Detection
    dst = cv2.cornerHarris(gray,2,3,0,1)
    dst = cv2.dilate(dst,None)

    #Gefundene Eckpunkte visualisieren
    image[dst>0.01*dst.max()]=[0,0,255]

    #Ausgabebbild an Bilschirmgroesse anpassen
    user32 = ctypes.windll.user32
    screen_size = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
    image_resized = cv2.resize(image, screen_size)

    #Angepasstes Bild anzeigen
    cv2.imshow('Harris Corner Detection', image_resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def sift(path):

    # Einlesen des Bildes
    img = cv2.imread(path)

    # Erstellen des SIFT-Detektors
    sift = cv2.SIFT_create()

    # Berechnung der Keypoints und Deskriptoren
    keypoints, descriptors = sift.detectAndCompute(img, None)

    # Zeichnen der Keypoints
    img_with_keypoints = cv2.drawKeypoints(img, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    #Ausgabebild an Bildschirmgroesse anpassen
    user32 = ctypes.windll.user32
    screen_size = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
    image_resized = cv2.resize(img_with_keypoints, screen_size)

    # Angepasstes Bild anzeigen
    cv2.imshow("Keypoints", image_resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def surf(path):
    # Einlesen des Bildes
    img = cv2.imread(path)

    # Erstellen des SURF-Objekts
    surf = cv2.xfeatures2d.SURF_create()

    # Extraktion der Features und Descriptoren
    keypoints, descriptors = surf.detectAndCompute(img, None)

    # Zeichnen der Features auf dem Bild
    cv2.drawKeypoints(img, keypoints, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Ausgabebild an Bildschirmgroesse anpassen
    user32 = ctypes.windll.user32
    screen_size = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
    image_resized = cv2.resize(img, screen_size)

    # Angepasstes Bild anzeigen
    cv2.imshow("Keypoints", image_resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




def brief(path):
    # Laden des Bildes
    img = cv2.imread(path)

    # Erstellen des ORB-Feature-Detektors
    orb = cv2.ORB_create()

    # Finden von Keypoints mit ORB
    kp = orb.detect(img, None)

    # Definition der binären Testfunktion
    # Beispiel: 256 zufällige Paare von Pixelpositionen
    np.random.seed(0)
    pairs = np.random.choice(32 * 32, (256, 2), replace=False)
    test = lambda x, y: x > y

    # Berechnung der BRIEF-Deskriptoren
    descriptors = []
    for point in kp:
        # Extrahieren des 32x32-Patches um den Keypoint
        patch = img[int(point.pt[1]) - 16:int(point.pt[1]) + 16, int(point.pt[0]) - 16:int(point.pt[0]) + 16]

        # Ausführen der binären Tests auf dem Patch
        descriptor = []
        for pair in pairs:
            descriptor.append(test(patch.flatten()[pair[0]], patch.flatten()[pair[1]]))
        descriptors.append(np.asarray(descriptor, dtype=np.uint8))

    # Konvertieren der Deskriptoren in ein numpy-Array
    descriptors = np.array(descriptors)

    # Zuordnung von Keypoints und Deskriptoren
    matches = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE_HAMMING).match(descriptors, descriptors)

    # Zeichnen von Keypoints auf dem Bild
    img_with_kp = cv2.drawKeypoints(img, kp, None, color=(0, 255, 0), flags=0)

    # Ausgabebild an Bildschirmgroesse anpassen
    user32 = ctypes.windll.user32
    screen_size = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
    image_resized = cv2.resize(img_with_kp, screen_size)

    # Anzeigen des Bildes
    cv2.imshow('Bild mit Keypoints', image_resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def orb(path):
    # Laden des Bildes
    img = cv2.imread(path)

    # Erstellen des ORB-Feature-Detektors
    orb = cv2.ORB_create()

    # Schlüsselpunkte und Deskriptoren berechnen
    kp, des = orb.detectAndCompute(img, None)

    # Zeichnen der Schlüsselpunkte
    img_with_kp = cv2.drawKeypoints(img, kp, None, color=(255, 0, 0), flags=0)

    # Ausgabebild an Bildschirmgroesse anpassen
    user32 = ctypes.windll.user32
    screen_size = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
    image_resized = cv2.resize(img_with_kp, screen_size)

    # Anzeigen des Bildes
    cv2.imshow('Bild mit Keypoints', image_resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def imageMatching(path):

    # Schritt 1: Feature Extraction

    # Liste aller Dateien im Ordner
    file_list = os.listdir(path)

    #Liste aller Bilder für weitere Verarbeitung
    img_list = []
    img_paths = []

    # Schleife durch jede Datei im Ordner
    for file_name in file_list:
        # Wenn die Datei eine Bilddatei ist (JPEG, PNG, etc.)
        if file_name.endswith('.jpg') or file_name.endswith('.png') or file_name.endswith('.jpeg'):
            # Pfad zur Bilddatei
            img_path = os.path.join(path, file_name)
            # Öffnen Sie die Bilddatei mit OpenCV
            img = cv2.imread(img_path)
            # Verkleinern der Bilder, um die Rechenzeit zu reduzieren
            img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
            img_list.append(img)
            img_paths.append(img_path)

    # Schritt 2: Extrahieren der Merkmale
    sift = cv2.SIFT_create()  # SIFT-Extraktor initialisieren
    kp_list = []
    des_list = []
    for img in img_list:
        kp, des = sift.detectAndCompute(img, None)
        kp_list.append(kp)
        des_list.append(des)

    # Schritt 3: Feature Matching
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)  # Brute-Force Matcher initialisieren
    match_counts = np.zeros((len(img_paths), len(img_paths)))
    for i in range(len(img_paths)):
        for j in range(i + 1, len(img_paths)):
            matches = bf.match(des_list[i], des_list[j])
            match_counts[i][j] = len(matches)
            match_counts[j][i] = match_counts[i][j]

    # Schritt 4: Wahrscheinlichkeitsberechnung
    probability_matrix = np.zeros((len(img_paths), len(img_paths)))
    for i in range(len(img_paths)):
        for j in range(i + 1, len(img_paths)):
            if match_counts[i][
                j] > 10:  # Schwellenwert, um eine ausreichende Anzahl von Übereinstimmungen zu gewährleisten
                probability_matrix[i][j] = match_counts[i][j] / float(
                    len(kp_list[i]) + len(kp_list[j]) - match_counts[i][j])
                probability_matrix[j][i] = probability_matrix[i][j]

    # Schritt 5: Ergebnisbewertung
    threshold = 0.2  # Schwellenwert, um Bilder auszuwählen, die für die 3D-Rekonstruktion verwendet werden sollen
    selected_images = []
    for i in range(len(img_paths)):
        if np.sum(probability_matrix[i]) > threshold:
            selected_images.append(i)

    print("Die folgenden Bilder wurden ausgewählt: ", selected_images)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    #Aufruf der Harris Corner Detection Funktion
    harrisCornerDetection("./pictures/IMG_0914.jpeg")

    #Aufruf des SIFT-Algorithmus
    sift("./pictures/IMG_0914.jpeg")

    # Aufruf des SURF-Algorithmus
    #surf("./pictures/IMG_0914.jpeg")

    # Aufruf des BRIEF-Algorithmus
    brief("./pictures/IMG_0914.jpeg")

    # Aufruf des BRIEF-Algorithmus
    orb("./pictures/IMG_0914.jpeg")

    #Aufruf des Image Matchings
    imageMatching("./pictures/")

