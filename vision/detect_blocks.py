#!/usr/bin/env python3
import cv2
import numpy as np
import json
import os

from detect_board import sample_color, find_window, find_grid_bounds

# ——— CONFIG ———
ASSETS           = "assets"
CAPTURE_PATH     = "latest_capture.png"
UI_BG_SAMPLE     = os.path.join(ASSETS, "inverted_background.png")
GRID_BG_SAMPLE   = os.path.join(ASSETS, "inverted_empty_tile.png")
OUTPUT_JSON      = "next_blocks.json"

PREVIEW_PAD      = 10    # pixels to skip under board
HUE_TOL          = 10    # HSV tolerance for grid‑bg
SAT_TOL          = 60
VAL_TOL          = 60

DEBUG            = True  # set False to skip visualization

def detect_pieces(full_img, ui_roi, grid_roi):
    """
    Returns (grids, preview, mask, boxes)
      - grids: list of 0/1 matrices for each piece
      - preview: BGR image crop under the board
      - mask: binary mask of piece‑pixels
      - boxes: [(x,y,w,h),…] in preview‑coords
    """
    x0, y0, w0, h0 = ui_roi
    gx, gy, gs, _  = grid_roi

    # 1) crop the strip under the 8×8 board
    top    = int(y0 + gy + gs + PREVIEW_PAD)
    bottom = y0 + h0
    left   = x0
    right  = x0 + w0
    preview = full_img[top:bottom, left:right]
    if preview.size == 0:
        return [], preview, None, []

    # 2) mask out grid background color in HSV
    grid_bgr = cv2.imread(GRID_BG_SAMPLE)
    gh, gw   = grid_bgr.shape[:2]
    bg_px    = grid_bgr[gh//2, gw//2]
    bg_hsv   = cv2.cvtColor(bg_px[None,None,:], cv2.COLOR_BGR2HSV)[0,0]

    prev_hsv = cv2.cvtColor(preview, cv2.COLOR_BGR2HSV)
    lower    = np.array([bg_hsv[0]-HUE_TOL, bg_hsv[1]-SAT_TOL, bg_hsv[2]-VAL_TOL])
    upper    = np.array([bg_hsv[0]+HUE_TOL, bg_hsv[1]+SAT_TOL, bg_hsv[2]+VAL_TOL])
    lower    = np.maximum(lower, 0)
    upper    = np.minimum(upper, [179,255,255])

    bg_mask  = cv2.inRange(prev_hsv, lower, upper)
    piece_mask = cv2.bitwise_not(bg_mask)
    piece_mask = cv2.medianBlur(piece_mask, 5)

    # 3) connected components → one blob per piece
    n, labels, stats, _ = cv2.connectedComponentsWithStats(piece_mask, connectivity=8)
    grids, boxes = [], []

    for lbl in range(1, n):
        area = stats[lbl, cv2.CC_STAT_AREA]
        # ignore tiny noise
        if area < 4:
            continue

        x = stats[lbl, cv2.CC_STAT_LEFT]
        y = stats[lbl, cv2.CC_STAT_TOP]
        w = stats[lbl, cv2.CC_STAT_WIDTH]
        h = stats[lbl, cv2.CC_STAT_HEIGHT]

        blob = piece_mask[y:y+h, x:x+w]

        # 4) infer tile size from blob area: avg square ≈ sqrt(area/4)
        avg_sq = np.sqrt(area / 4.0)
        # rows ≈ h/avg_sq, cols ≈ w/avg_sq
        rows = max(1, int(round(h/avg_sq)))
        cols = max(1, int(round(w/avg_sq)))

        # clamp to reasonable tetromino dims
        rows = min(rows, 4)
        cols = min(cols, 4)

        # 5) build 0/1 occupancy grid
        grid = []
        for i in range(rows):
            r = []
            for j in range(cols):
                x1 = int(j * w/cols)
                y1 = int(i * h/rows)
                x2 = int((j+1) * w/cols)
                y2 = int((i+1) * h/rows)
                cell = blob[y1:y2, x1:x2]
                r.append(1 if cell.mean() > 0.3 else 0)
            grid.append(r)

        grids.append(grid)
        boxes.append((x,y,w,h))

    # 6) sort left→right by box x
    order = sorted(range(len(boxes)), key=lambda i: boxes[i][0])
    grids = [grids[i] for i in order]

    return grids, preview, piece_mask, boxes


def visualize(full, ui_roi, grid_roi, board_mat, preview, mask, boxes):
    vis = full.copy()
    x0, y0, _, _ = ui_roi
    gx, gy, gs, _ = grid_roi

    # draw board grid overlay
    cv2.rectangle(vis, (x0, y0), (x0+ui_roi[2], y0+ui_roi[3]), (255,0,0), 2)
    cv2.rectangle(vis,
        (x0+gx, y0+gy),
        (x0+gx+gs, y0+gy+gs),
        (0,255,255), 2)

    cell = gs // 8
    for i, row in enumerate(board_mat):
        for j, val in enumerate(row):
            c = (0,0,255) if val else (0,255,0)
            x1 = x0 + gx + j*cell
            y1 = y0 + gy + i*cell
            cv2.rectangle(vis, (x1,y1), (x1+cell,y1+cell), c, 1)

    # tint detected pieces
    top  = int(y0 + gy + gs + PREVIEW_PAD)
    left = x0
    for (x,y,w,h) in boxes:
        cv2.rectangle(vis, (left+x, top+y), (left+x+w, top+y+h), (0,165,255), 2)

    cv2.imshow("All Detection", vis)
    cv2.imshow("Preview", preview)
    cv2.imshow("Mask", mask*255)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    full    = cv2.imread(CAPTURE_PATH)
    ui_bg   = sample_color(UI_BG_SAMPLE)
    grid_bg = sample_color(GRID_BG_SAMPLE)

    # 1) find window & board
    ui_roi   = find_window(full, ui_bg)
    win      = full[ui_roi[1]:ui_roi[1]+ui_roi[3],
                    ui_roi[0]:ui_roi[0]+ui_roi[2]]
    grid_roi = find_grid_bounds(win, grid_bg)

    # 2) board matrix
    board_mat = extract_matrix(win, grid_roi, grid_bg)
    with open(BOARD_JSON, "w") as f:
        json.dump(board_mat, f, indent=2)
    print("Board matrix:", board_mat)

    # 3) detect pieces
    grids, preview, mask, boxes = detect_pieces(full, ui_roi, grid_roi)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(grids, f, indent=2)
    print("Block matrices:", grids)

    # 4) visualize if desired
    if DEBUG:
        visualize(full, ui_roi, grid_roi, board_mat, preview, mask, boxes)


if __name__=="__main__":
    main()
