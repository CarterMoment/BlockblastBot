#!/usr/bin/env python3
import cv2, numpy as np, json, os
from detect_board import (
    sample_color,
    find_window,
    find_grid_bounds,
    extract_matrix
)

# ——— CONFIGURATION ———
ASSETS         = "assets"
CAPTURE_PATH   = "latest_capture.png"
UI_BG_SAMPLE   = os.path.join(ASSETS, "inverted_background.png")
GRID_SAMPLE    = os.path.join(ASSETS, "inverted_empty_tile.png")
BOARD_JSON     = "board_matrix.json"
BLOCKS_JSON    = "block_list.json"

# HSV tolerances for masking out the board‐background
HUE_TOL        = 15
SAT_TOL        = 60
VAL_TOL        = 80

# Preview‐strip padding (to avoid UI chrome)
PREVIEW_PAD    = 5

# Minimum contour area (in px²) to count as a block
MIN_BLOCK_AREA = 200

DEBUG          = True

def detect_blocks(full, ui_roi, grid_roi, grid_bg_bgr):
    """Return list of (x,y,w,h) for each bottom‐preview block."""
    x0,y0,w0,h0 = ui_roi
    gx,gy,gs,_  = grid_roi

    # 1) Crop the region *below* the 8×8 board
    top    = y0 + gy + gs + PREVIEW_PAD
    bottom = y0 + h0 - PREVIEW_PAD
    left   = x0 + PREVIEW_PAD
    right  = x0 + w0 - PREVIEW_PAD

    preview = full[top:bottom, left:right]
    if preview.size == 0:
        return [], preview, None

    # 2) Mask out the grid‐background color in HSV
    hsv        = cv2.cvtColor(preview, cv2.COLOR_BGR2HSV)
    bg_hsv     = cv2.cvtColor(
                    np.uint8([[grid_bg_bgr]]),
                    cv2.COLOR_BGR2HSV
                 )[0,0]
    lower = np.array([
        max(0,   bg_hsv[0] - HUE_TOL),
        max(0,   bg_hsv[1] - SAT_TOL),
        max(0,   bg_hsv[2] - VAL_TOL),
    ])
    upper = np.array([
        min(179, bg_hsv[0] + HUE_TOL),
        min(255, bg_hsv[1] + SAT_TOL),
        min(255, bg_hsv[2] + VAL_TOL),
    ])
    bg_mask    = cv2.inRange(hsv, lower, upper)
    block_mask = cv2.bitwise_not(bg_mask)
    block_mask = cv2.medianBlur(block_mask, 7)

    # 3) Find contours → each is one preview block
    cnts,_ = cv2.findContours(block_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        if w*h >= MIN_BLOCK_AREA:
            boxes.append((x,y,w,h))

    # 4) shift box coords back to full‐image space
    boxes_full = [(left+x, top+y, w, h) for x,y,w,h in boxes]
    return boxes_full, preview, block_mask

def visualize_all(full, ui_roi, grid_roi, matrix, boxes):
    x0,y0,w0,h0 = ui_roi
    gx,gy,gs,_  = grid_roi

    vis = full.copy()

    # draw window & board
    cv2.rectangle(vis, (x0,y0), (x0+w0,y0+h0), (255,0,0), 2)
    cv2.rectangle(vis,
                  (x0+gx,y0+gy),
                  (x0+gx+gs, y0+gy+gs),
                  (0,255,255), 2)

    # overlay the 8×8 matrix
    cell = gs // len(matrix)
    for i,row in enumerate(matrix):
        for j,val in enumerate(row):
            color = (0,0,255) if val else (0,255,0)
            x1 = x0 + gx + j*cell
            y1 = y0 + gy + i*cell
            cv2.rectangle(vis, (x1,y1),
                          (x1+cell, y1+cell),
                          color, 1)

    # draw bottom‐block boxes
    for (bx,by,bw,bh) in boxes:
        cv2.rectangle(vis, (bx,by), (bx+bw, by+bh), (0,165,255), 2)

    cv2.imshow("All Detection", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    full    = cv2.imread(CAPTURE_PATH)
    ui_bg   = sample_color(UI_BG_SAMPLE)
    grid_bg = sample_color(GRID_SAMPLE)

    # 1) Locate window and board
    ui_roi   = find_window(full, ui_bg)
    win_img  = full[ui_roi[1]:ui_roi[1]+ui_roi[3],
                    ui_roi[0]:ui_roi[0]+ui_roi[2]]
    grid_roi = find_grid_bounds(win_img, grid_bg)

    # 2) Extract the 8×8 matrix
    matrix = extract_matrix(win_img, grid_roi, grid_bg)

    # 3) Save board matrix
    with open(BOARD_JSON, "w") as f:
        json.dump(matrix, f, indent=2)

    # 4) Detect bottom blocks
    boxes, preview, mask = detect_blocks(full, ui_roi, grid_roi, grid_bg)

    # 5) Save block list
    with open(BLOCKS_JSON, "w") as f:
        json.dump(boxes, f, indent=2)

    print("Board matrix:")
    for r in matrix:
        print(r)
    print("Detected blocks (x,y,w,h):", boxes)

    # 6) Visualize everything
    if DEBUG:
        visualize_all(full, ui_roi, grid_roi, matrix, boxes)
        # optional: also show preview + mask
        cv2.imshow("Preview", preview)
        cv2.imshow("Mask", mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__=="__main__":
    main()
