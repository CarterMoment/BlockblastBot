import cv2
import numpy as np
from PIL import ImageGrab
import time
from capture_quicktime_utils import get_hsv_from_background  # Keep as is

# === CONFIGURATION ===
PADDING = 10
REFRESH_DELAY = 1.5  # in seconds
SAMPLE_COORD = (10, 10)  # Coordinates inside blank background image
CAPTURE_PATH = "latest_capture.png"

def live_capture_blockblast_window(hsv_color, hue_range=15, sat_range=60, val_range=80):
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
    print(f"[INFO] HSV range: {lower} to {upper}")

    print("[INFO] Starting capture loop. Press 'q' to stop.")
    while True:
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
            x1, y1 = max(x - PADDING, 0), max(y - PADDING, 0)
            x2, y2 = min(x + w + PADDING, screen_np.shape[1]), min(y + h + PADDING, screen_np.shape[0])
            region = (x1, y1, x2, y2)

            # Save to disk
            cropped = screen_np[y1:y2, x1:x2]
            cv2.imwrite(CAPTURE_PATH, cropped)
            print(f"[INFO] Saved capture to {CAPTURE_PATH}")

            # Optional: show it
            cv2.imshow("Block Blast Detected", cropped)
        else:
            print("[WARN] No matching region found.")
            cv2.imshow("Block Blast Detected", np.zeros((100, 100, 3), dtype=np.uint8))

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        time.sleep(REFRESH_DELAY)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    hsv_color = get_hsv_from_background("assets/blank_background.png", sample_coord=SAMPLE_COORD)
    live_capture_blockblast_window(hsv_color)
