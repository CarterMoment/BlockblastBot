import cv2
import numpy as np

def extract_board_matrix(image_input, debug=False):
    # Accept image directly (NumPy array)
    img = cv2.imread(image_input) if isinstance(image_input, str) else image_input
    h, w = img.shape[:2]

    # Crop region inside the captured window that corresponds to the board
    # These ratios are based on YOUR window dimensions (like 500x700)
    # Tune if needed for tighter fit
    top = int(0.07 * h)     # around the board top
    bottom = int(0.63 * h)  # just before bottom blocks
    left = int(0.10 * w)
    right = int(0.90 * w)

    board_img = img[top:bottom, left:right]
    bh, bw = board_img.shape[:2]
    cell_height = bh // 8
    cell_width = bw // 8

    board_matrix = []

    for row in range(8):
        row_vals = []
        for col in range(8):
            cell = board_img[
                row * cell_height : (row + 1) * cell_height,
                col * cell_width : (col + 1) * cell_width,
            ]
            gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
            mean_val = np.mean(gray)
            row_vals.append(1 if mean_val > 50 else 0)
        board_matrix.append(row_vals)

    if debug:
        for row in board_matrix:
            print(row)

    return board_matrix, board_img