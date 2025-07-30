#!/usr/bin/env python3
import cv2, numpy as np, json, os
from detect_board import find_window, find_grid_bounds, extract_matrix, sample_color

# ——— CONFIG ———
ASSETS         = "assets"
SHOT           = "latest_capture.png"
GRID_SAMPLE    = os.path.join(ASSETS, "inverted_empty_tile.png")
OUT_JSON       = "next_blocks.json"

# how much to pad under the board before detecting pieces
PREVIEW_PAD    = 10  

# HSV tolerances to mask away board background
HUE_TOL        = 10
SAT_TOL        = 60
VAL_TOL        = 60

# drop very small blobs
MIN_AREA_FRAC  = 0.2   # 20% of one tile area

# debug visualizations?
DEBUG          = True

def detect_blocks(full_img, ui_roi, grid_roi):
    x0,y0,w0,h0 = ui_roi
    gx,gy,gs,_  = grid_roi
    tile_px     = gs / 8.0

    # 1) Crop just under the board
    top    = int(y0 + gy + gs + PREVIEW_PAD)
    bottom = y0 + h0
    left   = x0
    right  = x0 + w0
    prev = full_img[top:bottom, left:right]
    if prev.size == 0:
        return []

    # 2) HSV‑mask out the board’s dark background
    grid_bgr = cv2.imread(GRID_SAMPLE)
    gh,gw    = grid_bgr.shape[:2]
    bg_px    = grid_bgr[gh//2, gw//2]
    bg_hsv   = cv2.cvtColor(bg_px[None,None,:], cv2.COLOR_BGR2HSV)[0,0]
    prev_hsv = cv2.cvtColor(prev, cv2.COLOR_BGR2HSV)
    lower = np.array([
        max(0,   bg_hsv[0]-HUE_TOL),
        max(0,   bg_hsv[1]-SAT_TOL),
        max(0,   bg_hsv[2]-VAL_TOL),
    ])
    upper = np.array([
        min(179, bg_hsv[0]+HUE_TOL),
        min(255, bg_hsv[1]+SAT_TOL),
        min(255, bg_hsv[2]+VAL_TOL),
    ])
    bg_mask    = cv2.inRange(prev_hsv, lower, upper)
    piece_mask = cv2.bitwise_not(bg_mask)
    piece_mask = cv2.medianBlur(piece_mask, 5)

    # 3) Connected components → each blob is one piece
    n, labels, stats, _ = cv2.connectedComponentsWithStats(piece_mask, connectivity=8)
    tile_area = tile_px * tile_px
    grids = []

    for lbl in range(1, n):
        area = stats[lbl, cv2.CC_STAT_AREA]
        if area < MIN_AREA_FRAC * tile_area:
            continue

        x = stats[lbl, cv2.CC_STAT_LEFT]
        y = stats[lbl, cv2.CC_STAT_TOP]
        w = stats[lbl, cv2.CC_STAT_WIDTH]
        h = stats[lbl, cv2.CC_STAT_HEIGHT]

        # 4) Subdivide blob bbox into rows×cols grid
        cols = max(1, int(round(w / tile_px)))
        rows = max(1, int(round(h / tile_px)))

        grid = []
        for i in range(rows):
            row = []
            for j in range(cols):
                x1 = int(x + j*tile_px)
                y1 = int(y + i*tile_px)
                x2 = int(x1 + tile_px)
                y2 = int(y1 + tile_px)
                cell = piece_mask[y1:y2, x1:x2]
                row.append(1 if cell.mean() > 0.2 else 0)
            grid.append(row)

        grids.append(grid)

    # sort pieces left→right
    # compute each blob center x for sorting
    centers = [
        (stats[l,cv2.CC_STAT_LEFT] + stats[l,cv2.CC_STAT_WIDTH]/2, l)
        for l in range(1,n)
        if stats[l,cv2.CC_STAT_AREA] >= MIN_AREA_FRAC * tile_area
    ]
    centers.sort(key=lambda x: x[0])
    sorted_grids = []
    for _, lbl in centers:
        # map lbl to corresponding grid in same order discovered
        # assume discovery order matches stats index order
        idx = [i for i,g in enumerate(grids) if True][lbl-1]
        sorted_grids.append(grids[idx])

    return sorted_grids[:3], prev, piece_mask

def visualize(full, ui_roi, grid_roi, matrix, preview, mask):
    x0,y0,w0,h0 = ui_roi
    gx,gy,gs,_  = grid_roi
    vis = full.copy()

    # draw board overlay
    cv2.rectangle(vis, (x0,y0), (x0+w0,y0+h0), (255,0,0),2)
    cv2.rectangle(vis,
        (x0+gx,y0+gy),(x0+gx+gs,y0+gy+gs),
        (0,255,255),2)

    # draw matrix
    cell = gs//8
    for i in range(8):
        for j in range(8):
            c = (0,0,255) if matrix[i][j] else (0,255,0)
            x1 = x0 + gx + j*cell
            y1 = y0 + gy + i*cell
            cv2.rectangle(vis,(x1,y1),(x1+cell,y1+cell),c,1)

    # draw detected piece masks in the preview region
    # shift mask into full-image coords
    top    = int(y0 + gy + gs + PREVIEW_PAD)
    left   = x0
    h_m, w_m = mask.shape
    overlay = vis[top:top+h_m, left:left+w_m]
    overlay[mask>0] = (0,165,255)  # tint colored blocks orange
    vis[top:top+h_m, left:left+w_m] = overlay

    cv2.imshow("All Detection", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    full     = cv2.imread(SHOT)
    ui_bg    = sample_color(UI_BG_SAMPLE)
    grid_bg  = sample_color(GRID_SAMPLE)

    # 1) locate window and board
    ui_roi   = find_window(full, ui_bg)
    win      = full[ui_roi[1]:ui_roi[1]+ui_roi[3],
                    ui_roi[0]:ui_roi[0]+ui_roi[2]]
    grid_roi = find_grid_bounds(win, grid_bg)

    # 2) extract the board matrix
    matrix = extract_matrix(win, grid_roi, grid_bg)
    with open(OUT_JSON, "w") as f:
        json.dump(matrix, f, indent=2)
    print("Board matrix:", matrix)

    # 3) detect bottom blocks as grids
    block_grids, prev, mask = detect_blocks(full, ui_roi, grid_roi)
    with open(OUT_JSON.replace("board","next_blocks"), "w") as f:
        json.dump(block_grids, f, indent=2)
    print("Block matrices:", block_grids)

    # 4) visualize
    if DEBUG:
        visualize(full, ui_roi, grid_roi, matrix, prev, mask)

if __name__=="__main__":
    main()
