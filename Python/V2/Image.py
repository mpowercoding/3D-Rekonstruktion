import PIL
import cv2


class Image:
    _img = None
    _camera_matrix = None
    _imagepath = None
    _dist_coeffs = None
    _features = None
    _position = None
    _connections = {}

    def __init__(self, imagepath, camera_matrix, dist_coeffs):
        self._imagepath = imagepath
        self._camera_matrix = camera_matrix
        self._dist_coeffs = dist_coeffs

    def get_image(self):
        if self._img is None:
            self._img = cv2.imread(self._imagepath, cv2.IMREAD_GRAYSCALE)
        return self._img

    def get_camera_matrix(self):
        return self._camera_matrix

    def get_dist_coeffs(self):
        return self._dist_coeffs

    def get_features(self):
        if self._features is None:
            sift = cv2.SIFT_create()
            kp, des = sift.detectAndCompute(self.get_image(), None)
            self._features = (kp, des)
        return self._features

    def set_postition(self, pos):
        self._position = pos

    def get_position(self):
        return self._position

    def get_connections(self):
        return self._connections

    def set_connections(self, json_obj):
        self._connections = json_obj

    def get_image_path(self):
        return self._imagepath

    @staticmethod
    def get_focal_length(path):
        from PIL import Image, ExifTags
        img = PIL.Image.open(path)
        exif = {ExifTags.TAGS[k]: v for k, v in img._getexif().items() if k in ExifTags.TAGS}
        return exif['FocalLength']
