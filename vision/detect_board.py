import cv2
import numpy as np

# === CONFIGURABLE CROPPING PARAMETERS (adjust after testing)
GRID_TOP_LEFT = (100, 150)  # x, y pixel offset from top-left corner of capture
GRID_SIZE = 560             # width and height of the board square in pixels
GRID_DIM = 8                # 8x8 Block Blast board
THRESHOLD = 50              # Brightness threshold for detecting "filled"

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

            # Convert to grayscale and check average brightness
            gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
            avg_brightness = np.mean(gray)
            filled = int(avg_brightness < THRESHOLD)  # dark = filled
            matrix_row.append(filled)
        matrix.append(matrix_row)

    return matrix

if __name__ == "__main__":
    matrix = extract_board_matrix("latest_capture.png")
    for row in matrix:
        print(row)
