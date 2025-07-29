import cv2
import numpy as np
import json
from vision.capture_quicktime_utils import get_hsv_from_background

# Config
GRID_SIZE = 8
BG_IMAGE_PATH = "assets/inverted_background.png"
EMPTY_TILE_PATH = "assets/inverted_empty_tile.png"
CAPTURE_PATH = "latest_capture.png"
SAMPLE_COORD = (10, 10)
TOLERANCE = np.array([15, 60, 80])  # H, S, V tolerance

def find_board_bounds(image_hsv, bg_hsv):
    lower = np.maximum(bg_hsv - TOLERANCE, 0)
    upper = np.minimum(bg_hsv + TOLERANCE, 255)
    mask = cv2.inRange(image_hsv, lower, upper)
    mask = cv2.bitwise_not(mask)  # Invert: we want non-bg
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("No board-like contour found.")
    return cv2.boundingRect(max(contours, key=cv2.contourArea))  # x, y, w, h

def extract_matrix(image_hsv, board_rect, empty_tile_hsv):
    x, y, w, h = board_rect
    cell_w, cell_h = w // GRID_SIZE, h // GRID_SIZE
    matrix = []

    for i in range(GRID_SIZE):
        row = []
        for j in range(GRID_SIZE):
            cx = x + j * cell_w + cell_w // 2
            cy = y + i * cell_h + cell_h // 2
            sample = image_hsv[cy, cx]
            diff = np.abs(sample.astype(int) - empty_tile_hsv.astype(int))
            row.append(0 if np.all(diff < TOLERANCE) else 1)
        matrix.append(row)

    return matrix

if __name__ == "__main__":
    full_rgb = cv2.imread(CAPTURE_PATH)
    full_hsv = cv2.cvtColor(full_rgb, cv2.COLOR_BGR2HSV)

    bg_hsv = get_hsv_from_background(BG_IMAGE_PATH, sample_coord=SAMPLE_COORD)
    empty_tile_hsv = get_hsv_from_background(EMPTY_TILE_PATH, sample_coord=SAMPLE_COORD)

    board_rect = find_board_bounds(full_hsv, bg_hsv)
    matrix = extract_matrix(full_hsv, board_rect, empty_tile_hsv)

    with open("board_matrix.json", "w") as f:
        json.dump(matrix, f)

    for row in matrix:
        print(row)
