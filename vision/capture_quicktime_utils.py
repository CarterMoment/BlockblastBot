import cv2
import numpy as np

def get_hsv_from_background(image_path, sample_coord=(10, 10)):
    image = cv2.imread(image_path)  # Loads as BGR
    if image is None:
        raise FileNotFoundError(f"Could not read: {image_path}")
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return hsv[sample_coord[1], sample_coord[0]]
