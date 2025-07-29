#!/usr/bin/env python3
import cv2
import numpy as np
import json
import argparse

def find_board_region(img, board_thresh=60, morph_size=15):
    """
    Find the largest dark region (the 8×8 grid) in the image.
    - board_thresh: grayscale threshold; pixels darker than this are considered grid.
    - morph_size: size of morphological closing kernel to fill small gaps.
    Returns (x, y, w, h) of a square crop around the board.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Mask darker pixels (the board background)
    _, mask = cv2.threshold(gray, board_thresh, 255, cv2.THRESH_BINARY_INV)
    # Close small holes to get one solid blob
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_size, morph_size))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # Find contours and pick the largest
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("Could not find the board region.")
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    # Make it square by taking the smaller side
    side = min(w, h)
    return x, y, side, side

def extract_board_matrix(img, region, grid_size=8, block_thresh=100):
    """
    Given the full image and the board region (x,y,w,h), split into grid_size^2 cells,
    compute each cell's mean gray value, and classify:
       mean_gray > block_thresh → 1 (filled), else 0 (empty).
    Returns an 8×8 list of lists.
    """
    x, y, w, h = region
    board = img[y:y+h, x:x+w]
    gray = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
    cell_h, cell_w = h // grid_size, w // grid_size

    matrix = []
    for i in range(grid_size):
        row = []
        for j in range(grid_size):
            cy1, cy2 = i*cell_h, (i+1)*cell_h
            cx1, cx2 = j*cell_w, (j+1)*cell_w
            cell = gray[cy1:cy2, cx1:cx2]
            mean_val = int(np.mean(cell))
            row.append(1 if mean_val > block_thresh else 0)
        matrix.append(row)
    return matrix

def main():
    p = argparse.ArgumentParser(description="Detect 8×8 Block Blast board from a screenshot.")
    p.add_argument("image", help="Path to the screenshot (e.g. latest_capture.png)")
    p.add_argument("-o","--output", default="board_matrix.json", help="Where to save the JSON matrix")
    p.add_argument("--board-thresh", type=int, default=60,
                   help="Gray threshold to find the board background (darker than this).")
    p.add_argument("--block-thresh", type=int, default=100,
                   help="Gray threshold to detect a filled block (brighter than this).")
    p.add_argument("--morph-size", type=int, default=15,
                   help="Kernel size for closing the board mask.")
    args = p.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        p.error(f"Could not load image: {args.image}")

    region = find_board_region(img, board_thresh=args.board_thresh, morph_size=args.morph_size)
    matrix = extract_board_matrix(img, region, grid_size=8, block_thresh=args.block_thresh)

    with open(args.output, "w") as f:
        json.dump(matrix, f, indent=2)
    print(f"✅ Saved board matrix to {args.output}")

if __name__ == "__main__":
    main()
