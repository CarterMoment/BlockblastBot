import cv2
import numpy as np
import json

IMG_PATH = "assets/latest_capture.png"
OUTPUT_PATH = "board_matrix.json"
GRID_SIZE = 8


def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blur, 50, 150, apertureSize=3)
    return edges


def detect_grid_lines(edges):
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    vertical_lines = cv2.erode(edges, vertical_kernel, iterations=1)
    vertical_lines = cv2.dilate(vertical_lines, vertical_kernel, iterations=1)

    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    horizontal_lines = cv2.erode(edges, horizontal_kernel, iterations=1)
    horizontal_lines = cv2.dilate(horizontal_lines, horizontal_kernel, iterations=1)

    grid_mask = cv2.addWeighted(vertical_lines, 0.5, horizontal_lines, 0.5, 0.0)
    return grid_mask


def find_board_contour(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No grid contour found.")
    return max(contours, key=cv2.contourArea)


def crop_to_board(img, contour):
    x, y, w, h = cv2.boundingRect(contour)
    return img[y:y+h, x:x+w]


def analyze_cells(cropped_img):
    h, w = cropped_img.shape[:2]
    tile_h, tile_w = h // GRID_SIZE, w // GRID_SIZE
    hsv_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2HSV)

    matrix = []

    for i in range(GRID_SIZE):
        row = []
        for j in range(GRID_SIZE):
            tile = hsv_img[i*tile_h:(i+1)*tile_h, j*tile_w:(j+1)*tile_w]
            avg_val = np.mean(tile[:, :, 2])  # V channel from HSV
            row.append(1 if avg_val < 100 else 0)  # Threshold — adjust if needed
        matrix.append(row)
    return matrix


def main():
    img = cv2.imread(IMG_PATH)
    if img is None:
        raise FileNotFoundError(f"Could not read {IMG_PATH}")

    edges = preprocess_image(img)
    grid_mask = detect_grid_lines(edges)
    contour = find_board_contour(grid_mask)
    board_img = crop_to_board(img, contour)
    matrix = analyze_cells(board_img)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(matrix, f, indent=2)

    print(f"[✅] Saved matrix to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
