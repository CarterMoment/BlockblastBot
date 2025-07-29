import cv2
import numpy as np
import json

# === CONFIGURABLE CROPPING PARAMETERS
GRID_TOP_LEFT = (100, 150)  # Adjust these
GRID_SIZE = 560
GRID_DIM = 8
THRESHOLD = 50
OUTPUT_JSON = "board_matrix.json"

def extract_board_matrix(image_path: str):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read {image_path}")

    x0, y0 = GRID_TOP_LEFT
    x1, y1 = x0 + GRID_SIZE, y0 + GRID_SIZE
    board_region = image[y0:y1, x0:x1]

    cell_size = GRID_SIZE // GRID_DIM
    matrix = []

    for row in range(GRID_DIM):
        matrix_row = []
        for col in range(GRID_DIM):
            cell_x1 = col * cell_size
            cell_y1 = row * cell_size
            cell = board_region[cell_y1:cell_y1+cell_size, cell_x1:cell_x1+cell_size]

            gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
            avg_brightness = np.mean(gray)
            filled = int(avg_brightness < THRESHOLD)  # filled if dark
            matrix_row.append(filled)
        matrix.append(matrix_row)

    return matrix

if __name__ == "__main__":
    matrix = extract_board_matrix("latest_capture.png")

    # Save to JSON
    with open(OUTPUT_JSON, "w") as f:
        json.dump(matrix, f)
    print(f"[INFO] Matrix saved to {OUTPUT_JSON}")
