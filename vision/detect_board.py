import cv2
import numpy as np
import json

# === Configuration
GRID_TOP_LEFT = (100, 150)        # Crop offset (adjust as needed)
GRID_SIZE = 560                   # Total pixel size of board square
GRID_DIM = 8                      # 8x8 Block Blast board
CELL_SIZE = GRID_SIZE // GRID_DIM
EMPTY_TILE_PATH = "assets/empty_tile.png"
COLOR_TOLERANCE = 30              # Color difference tolerance
OUTPUT_JSON = "board_matrix.json"

def get_empty_tile_color(path: str) -> np.ndarray:
    tile_img = cv2.imread(path)
    if tile_img is None:
        raise FileNotFoundError(f"Could not read {path}")
    h, w, _ = tile_img.shape
    center_color = tile_img[h // 2, w // 2]
    print(f"[INFO] Empty tile color (BGR): {center_color}")
    return center_color

def is_empty_cell(cell_img: np.ndarray, empty_color: np.ndarray) -> bool:
    center = cell_img[cell_img.shape[0] // 2, cell_img.shape[1] // 2]
    diff = np.abs(center.astype(int) - empty_color.astype(int))
    return np.all(diff < COLOR_TOLERANCE)

def extract_board_matrix(image_path: str, empty_color: np.ndarray):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read {image_path}")

    x0, y0 = GRID_TOP_LEFT
    board = image[y0:y0 + GRID_SIZE, x0:x0 + GRID_SIZE]

    matrix = []
    for row in range(GRID_DIM):
        matrix_row = []
        for col in range(GRID_DIM):
            x1 = col * CELL_SIZE
            y1 = row * CELL_SIZE
            cell = board[y1:y1 + CELL_SIZE, x1:x1 + CELL_SIZE]

            empty = is_empty_cell(cell, empty_color)
            matrix_row.append(0 if empty else 1)
        matrix.append(matrix_row)

    return matrix

if __name__ == "__main__":
    empty_color = get_empty_tile_color(EMPTY_TILE_PATH)
    matrix = extract_board_matrix("latest_capture.png", empty_color)

    with open(OUTPUT_JSON, "w") as f:
        json.dump(matrix, f)
    print(f"[INFO] Matrix saved to {OUTPUT_JSON}")
