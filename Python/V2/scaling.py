import cv2


def downsample_image(image, reduce_factor):
    for i in range(0, reduce_factor):
        if len(image.shape) > 2:
            row, col = image.shape[:2]
        else:
            row, col = image.shape
        image = cv2.pyrDown(image, dstsize=(col // 2, row // 2))
    return image
