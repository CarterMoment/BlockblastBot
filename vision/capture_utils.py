import cv2
import numpy as np
from PIL import ImageGrab

PADDING = 10

def get_blockblast_frame(hsv_color, hue_range=15, sat_range=60, val_range=80):
    screen = ImageGrab.grab()
    screen_np = np.array(screen)
    screen_bgr = cv2.cvtColor(screen_np, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(screen_bgr, cv2.COLOR_BGR2HSV)

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

    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.medianBlur(mask, 7)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        x1, y1 = max(x - PADDING, 0), max(y - PADDING, 0)
        x2, y2 = min(x + w + PADDING, screen_np.shape[1]), min(y + h + PADDING, screen_np.shape[0])
        cropped = screen_bgr[y1:y2, x1:x2]
        return cropped
    else:
        return None
