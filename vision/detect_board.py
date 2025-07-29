#!/usr/bin/env python3
import cv2
import numpy as np
import json

# === CONFIGURATION ===
IMG_PATH        = "latest_capture.png"
EMPTY_TILE_PATH = "assets/inverted_empty_tile.png"
OUTPUT_JSON     = "board_matrix.json"
GRID_SIZE       = 8
HSV_TOLERANCE   = np.array([10, 60, 60])   # H, S, V tolerances

def load_hsv(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read: {path}")
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, w = hsv.shape[:2]
    return hsv[h//2, w//2]

def find_board_region(img, thresh=60, morph=15):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, m = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph, morph))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    side = min(w, h)
    return x, y, side, side

def extract_board_matrix(img, empty_hsv):
    x, y, w, h = find_board_region(img)
    tile_w, tile_h = w // GRID_SIZE, h // GRID_SIZE
    hsv_full = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    matrix = []
    for i in range(GRID_SIZE):
        row = []
        for j in range(GRID_SIZE):
            # sample a 50%×50% centered patch of each cell
            x1 = x + j*tile_w + tile_w//4
            y1 = y + i*tile_h + tile_h//4
            x2 = x + j*tile_w + 3*tile_w//4
            y2 = y + i*tile_h + 3*tile_h//4
            patch = hsv_full[y1:y2, x1:x2]

            avg_hsv = patch.reshape(-1,3).mean(axis=0)
            diff = np.abs(avg_hsv - empty_hsv)
            cell = 0 if np.all(diff < HSV_TOLERANCE) else 1
            row.append(cell)
        matrix.append(row)
    return matrix

def main():
    img = cv2.imread(IMG_PATH)
    if img is None:
        raise FileNotFoundError(f"Could not load screenshot: {IMG_PATH}")

    empty_hsv = load_hsv(EMPTY_TILE_PATH)
    board = extract_board_matrix(img, empty_hsv)

    with open(OUTPUT_JSON, "w") as f:
        json.dump(board, f, indent=2)

    print(f"✅ Saved board matrix to {OUTPUT_JSON}")
    for row in board:
        print(row)

if __name__ == "__main__":
    main()
