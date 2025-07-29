from PIL import Image
import cv2
import numpy as np

def get_hsv_from_background(path="../assets/blank_background.png", sample_coord=(10, 10)):
    img = Image.open(path).convert("RGB")
    r, g, b = img.getpixel(sample_coord)
    rgb = np.uint8([[[r, g, b]]])
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[0][0]
    print(f"[INFO] Sampled HSV at {sample_coord}: {hsv}")
    return hsv

    