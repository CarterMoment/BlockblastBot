import cv2
import numpy as np
from PIL import ImageGrab, Image
import time

def get_blockblast_window(crown_path="assets/crown_anchor.png", sample_offset=(0, 80), hue_range=10, sat_range=50, val_range=60):
    # Take a full screenshot
    screenshot = ImageGrab.grab()
    screenshot_np = np.array(screenshot)
    screenshot_bgr = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2BGR)

    # Load the anchor (crown icon)
    anchor = cv2.imread(crown_path)
    if anchor is None:
        print("[ERROR] Could not load anchor image.")
        return None

    # Match the anchor in the full screen
    result = cv2.matchTemplate(screenshot_bgr, anchor, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    if max_val < 0.8:
        print("[WARN] Anchor not found on screen.")
        return None

    # Sample pixel near anchor to get background color
    anchor_x, anchor_y = max_loc
    sample_x = anchor_x + sample_offset[0]
    sample_y = anchor_y + sample_offset[1]

    sample_bgr = screenshot_bgr[sample_y, sample_x]
    sample_rgb = np.array([[sample_bgr[::-1]]], dtype=np.uint8)
    hsv_sample = cv2.cvtColor(sample_rgb, cv2.COLOR_RGB2HSV)[0][0]

    lower = np.array([
        max(0, hsv_sample[0] - hue_range),
        max(0, hsv_sample[1] - sat_range),
        max(0, hsv_sample[2] - val_range)
    ])
    upper = np.array([
        min(179, hsv_sample[0] + hue_range),
        min(255, hsv_sample[1] + sat_range),
        min(255, hsv_sample[2] + val_range)
    ])

    # Mask and extract the window
    hsv = cv2.cvtColor(screenshot_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.medianBlur(mask, 7)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        cropped = screenshot_bgr[y:y+h, x:x+w]
        return cropped
    else:
        print("[WARN] No region matching background color found.")
        return None

if __name__ == "__main__":
    print("[INFO] Starting Block Blast capture loop. Press 'q' to quit.")
    while True:
        frame = get_blockblast_window()
        if frame is not None:
            cv2.imshow("Block Blast Window", frame)
        else:
            black = np.zeros((100, 100, 3), dtype=np.uint8)
            cv2.imshow("Block Blast Window", black)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        time.sleep(0.3)

    cv2.destroyAllWindows()
