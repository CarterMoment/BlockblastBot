import cv2
import numpy as np
import json
import os

ASSETS = "assets"
CAPTURE_PATH = os.path.join(ASSETS, "latest_capture.png")
BG_SAMPLE_PATH = os.path.join(ASSETS, "inverted_background.png")
EMPTY_SAMPLE_PATH = os.path.join(ASSETS, "inverted_empty_tile.png")
OUTPUT_JSON = "board_matrix.json"

GRID_SIZE = 8
TILE_TOLERANCE = 30  # max HSV distance for a match


def load_hsv_sample(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read sample image: {path}")
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return hsv[0, 0]


def hsv_distance(c1, c2):
    return np.linalg.norm(np.array(c1, dtype=np.float32) - np.array(c2, dtype=np.float32))


def find_board_bounds(image, background_hsv):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower = np.clip(background_hsv - np.array([8, 40, 40]), 0, 255)
    upper = np.clip(background_hsv + np.array([8, 40, 40]), 0, 255)

    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    mask_inv = cv2.bitwise_not(mask)

    contours, _ = cv2.findContours(mask_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("Could not locate board region.")

    board_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(board_contour)
    return x, y, w, h


def build_board_matrix(capture_img, board_region, empty_tile_hsv):
    x, y, w, h = board_region
    tile_w = w // GRID_SIZE
    tile_h = h // GRID_SIZE

    board_matrix = []

    hsv_capture = cv2.cvtColor(capture_img, cv2.COLOR_BGR2HSV)

    for row in range(GRID_SIZE):
        row_values = []
        for col in range(GRID_SIZE):
            cx = x + col * tile_w + tile_w // 2
            cy = y + row * tile_h + tile_h // 2

            pixel_hsv = hsv_capture[cy, cx]
            dist = hsv_distance(pixel_hsv, empty_tile_hsv)

            if dist < TILE_TOLERANCE:
                row_values.append(0)
            else:
                row_values.append(1)
        board_matrix.append(row_values)

    return board_matrix


def main():
    capture_img = cv2.imread(CAPTURE_PATH)
    if capture_img is None:
        raise FileNotFoundError("Could not load game capture image.")

    background_hsv = load_hsv_sample(BG_SAMPLE_PATH)
    empty_tile_hsv = load_hsv_sample(EMPTY_SAMPLE_PATH)

    board_region = find_board_bounds(capture_img, background_hsv)
    board_matrix = build_board_matrix(capture_img, board_region, empty_tile_hsv)

    with open(OUTPUT_JSON, "w") as f:
        json.dump(board_matrix, f, indent=2)
    print(f"[✅] Saved board matrix to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
