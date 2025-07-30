#!/usr/bin/env python3
import cv2
import numpy as np
import json
import os

from detect_board import sample_color, find_window, find_grid_bounds, extract_matrix

# ——— CONFIGURATION ———
ASSETS           = "assets"
SHOT             = os.path.join(ASSETS, "latest_capture.png")
UI_BG_SAMPLE     = os.path.join(ASSETS, "inverted_background.png")
GRID_BG_SAMPLE   = os.path.join(ASSETS, "inverted_empty_tile.png")
BOARD_JSON       = "board_matrix.json"
BLOCKS_JSON      = "next_blocks.json"

# how many px above the board bottom to start
PREVIEW_OFFSET   = -5     
# how many tile‑heights to grab below that
STRIP_TILES      = 2.4    

# HSV tolerances for masking out grid background
HUE_TOL          = 10
SAT_TOL          = 60
VAL_TOL          = 60

DEBUG            = True   # set False to skip all cv2.imshow calls

def detect_pieces(full_img, ui_roi, grid_roi):
    x0, y0, w0, h0 = ui_roi
    gx, gy, gs, _  = grid_roi

    # 1) crop a narrow strip just under the board
    tile_h    = gs / 8.0
    top       = int(y0 + gy + gs + PREVIEW_OFFSET)
    preview_h = int(tile_h * STRIP_TILES)
    preview   = full_img[top: top + preview_h, x0: x0 + w0]
    if preview.size == 0:
        return [], preview, None, []

    # 2) mask out the board‑background color in HSV
    grid_bgr = cv2.imread(GRID_BG_SAMPLE)
    gh, gw   = grid_bgr.shape[:2]
    bg_px    = grid_bgr[gh//2, gw//2]
    bg_hsv   = cv2.cvtColor(bg_px[None,None,:], cv2.COLOR_BGR2HSV)[0,0]

    hsv       = cv2.cvtColor(preview, cv2.COLOR_BGR2HSV)
    lower     = np.array([bg_hsv[0]-HUE_TOL, bg_hsv[1]-SAT_TOL, bg_hsv[2]-VAL_TOL])
    upper     = np.array([bg_hsv[0]+HUE_TOL, bg_hsv[1]+SAT_TOL, bg_hsv[2]+VAL_TOL])
    lower     = np.clip(lower, [0,0,0], [179,255,255]).astype(np.uint8)
    upper     = np.clip(upper, [0,0,0], [179,255,255]).astype(np.uint8)

    bg_mask   = cv2.inRange(hsv, lower, upper)
    piece_mask= cv2.bitwise_not(bg_mask)
    piece_mask= cv2.medianBlur(piece_mask, 5)

    # 3) connected‑components → one blob per piece
    n, labels, stats, _ = cv2.connectedComponentsWithStats(piece_mask, connectivity=8)
    tile_px = gs / 8.0
    min_area= 0.2 * tile_px * tile_px

    grids, boxes = [], []
    for lbl in range(1, n):
        area = stats[lbl, cv2.CC_STAT_AREA]
        if area < min_area:
            continue

        x = stats[lbl, cv2.CC_STAT_LEFT]
        y = stats[lbl, cv2.CC_STAT_TOP]
        w = stats[lbl, cv2.CC_STAT_WIDTH]
        h = stats[lbl, cv2.CC_STAT_HEIGHT]
        blob = piece_mask[y:y+h, x:x+w]

        # 4) infer avg square size and thereby rows/cols
        avg_sq = np.sqrt(area / 4.0)
        rows   = max(1, min(4, int(round(h / avg_sq))))
        cols   = max(1, min(4, int(round(w / avg_sq))))

        # 5) subdivide blob into rows×cols occupancy
        grid = []
        for i in range(rows):
            row = []
            for j in range(cols):
                x1 = int(j * w/cols)
                y1 = int(i * h/rows)
                x2 = int((j+1) * w/cols)
                y2 = int((i+1) * h/rows)
                cell = blob[y1:y2, x1:x2]
                row.append(1 if cell.mean() > 0.3 else 0)
            grid.append(row)

        grids.append(grid)
        boxes.append((x, y, w, h))

    # sort left→right
    order = sorted(range(len(boxes)), key=lambda i: boxes[i][0])
    grids = [grids[i] for i in order]
    boxes=  [boxes[i] for i in order]

    return grids, preview, piece_mask, boxes

def visualize(full, ui_roi, grid_roi, board_mat, preview, mask, boxes):
    vis = full.copy()
    x0,y0,_,_ = ui_roi
    gx,gy,gs,_ = grid_roi

    # draw board and 8×8 overlay
    cv2.rectangle(vis, (x0,y0), (x0+ui_roi[2], y0+ui_roi[3]), (255,0,0), 2)
    cv2.rectangle(vis,
        (x0+gx, y0+gy),
        (x0+gx+gs, y0+gy+gs),
        (0,255,255), 2)

    cell = gs // 8
    for i,row in enumerate(board_mat):
        for j,val in enumerate(row):
            c = (0,0,255) if val else (0,255,0)
            x1 = x0 + gx + j*cell
            y1 = y0 + gy + i*cell
            cv2.rectangle(vis, (x1,y1), (x1+cell,y1+cell), c, 1)

    # outline each detected preview piece
    strip_top = int(y0 + gy + gs + PREVIEW_OFFSET)
    for (x,y,w,h) in boxes:
        cv2.rectangle(vis,
            (x0+x, strip_top+y),
            (x0+x+w, strip_top+y+h),
            (0,165,255), 2)

    cv2.imshow("All Detection", vis)
    cv2.imshow("Preview Strip", preview)
    cv2.imshow("Piece Mask", mask*255)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    full     = cv2.imread(SHOT)
    ui_bg    = sample_color(UI_BG_SAMPLE)
    grid_bg  = sample_color(GRID_BG_SAMPLE)

    # 1) detect board + extract 8×8 matrix
    ui_roi   = find_window(full, ui_bg)
    win      = full[ui_roi[1]:ui_roi[1]+ui_roi[3],
                    ui_roi[0]:ui_roi[0]+ui_roi[2]]
    grid_roi = find_grid_bounds(win, grid_bg)
    board_mat= extract_matrix(win, grid_roi, grid_bg)
    with open(BOARD_JSON, "w") as f:
        json.dump(board_mat, f, indent=2)
    print("Board matrix:", board_mat)

    # 2) detect bottom pieces → 0/1 grids
    grids, preview, mask, boxes = detect_pieces(full, ui_roi, grid_roi)
    with open(BLOCKS_JSON, "w") as f:
        json.dump(grids, f, indent=2)
    print("Next-block matrices:", grids)

    # 3) debug visualize
    if DEBUG:
        visualize(full, ui_roi, grid_roi, board_mat, preview, mask, boxes)

if __name__ == "__main__":
    main()
