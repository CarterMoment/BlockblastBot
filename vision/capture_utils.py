# capture_utils.py
import cv2
import numpy as np
from PIL import ImageGrab
from .capture_quicktime_utils import get_hsv_from_background  # or adjust if flat

def get_blockblast_frame_from_quicktime():
    hsv_color = get_hsv_from_background("assets/blank_background.png", sample_coord=(10, 10))
    hue_range, sat_range, val_range = 15, 60, 80

    lower = np.array([
        max(0, hsv_color[0] - hue_range),
        max(0, hsv_color[1] - sat_range),
        max(0, hsv_color[2] - val_range)
    ])
    upper = np.array([
        min(179, hsv_color[0] + hue_range),
        min(255, hsv_color[1] + sat_range),
        min(255, hsv_color[2] + val_range)
    ])

    screen = ImageGrab.grab()
    screen_np = np.array(screen)
    screen_bgr = cv2.cvtColor(screen_np, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(screen_bgr, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.medianBlur(mask, 7)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        cropped = screen_bgr[y:y + h, x:x + w]
        return cropped
    else:
        return None
