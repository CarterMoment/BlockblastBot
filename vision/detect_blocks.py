#!/usr/bin/env python3
import cv2
import numpy as np
import json
import os

from detect_board import find_window, find_grid_bounds, sample_color

# ——— CONFIG ———
ASSETS_DIR       = "assets"
CAPTURE_PATH     = os.path.join(ASSETS_DIR, "latest_capture.png")
SAMPLE_TILE_PATH = os.path.join(ASSETS_DIR, "sample_block.png")
OUTPUT_JSON      = "next_blocks.json"

PREVIEW_PAD      = 10      # px crop under the board
MASK_DELTA       = 30      # gray-difference tolerance
OCC_THRESH       = 0.4     # ≥40% of cell → occupied
MIN_AREA_FRAC    = 0.25    # drop tiny blobs < this fraction of tile²
DEBUG            = False

def detect_next_blocks(full, ui_roi, board_roi):
    x0,y0,w0,h0 = ui_roi
    bx,by,bs,_  = board_roi

    # 1) Crop the preview strip
    H,W = full.shape[:2]
    top    = min(H, y0+by+bs + PREVIEW_PAD)
    bottom = max(0, H - PREVIEW_PAD)
    left   = min(W, x0 + PREVIEW_PAD)
    right  = max(0, x0 + w0 - PREVIEW_PAD)
    if bottom<=top or right<=left:
        return [], None, None, None
    prev = full[top:bottom, left:right]
    gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    # 2) Threshold around sample tile’s median gray
    samp = cv2.imread(SAMPLE_TILE_PATH, cv2.IMREAD_GRAYSCALE)
    if samp is None:
        raise FileNotFoundError(f"Missing {SAMPLE_TILE_PATH}")
    med = int(np.median(samp))
    diff = cv2.absdiff(gray, np.full_like(gray, med))
    mask = (diff <= MASK_DELTA).astype(np.uint8)

    # 3) Morphology clean
    kern = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kern)

    # 4) Connected-components → one blob per piece
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    tile_px = bs/8.0
    grids = []
    boxes = []

    for lbl in range(1,n):
        area = stats[lbl,cv2.CC_STAT_AREA]
        if area < MIN_AREA_FRAC * (tile_px**2):
            continue

        x = stats[lbl,cv2.CC_STAT_LEFT]
        y = stats[lbl,cv2.CC_STAT_TOP]
        w = stats[lbl,cv2.CC_STAT_WIDTH]
        h = stats[lbl,cv2.CC_STAT_HEIGHT]

        # 5) Build an occupancy grid of size rows×cols
        cols = max(1, int(round(w / tile_px)))
        rows = max(1, int(round(h / tile_px)))
        grid = []
        for i in range(rows):
            row=[]
            for j in range(cols):
                x1 = int(x + j*tile_px)
                x2 = int(x + (j+1)*tile_px)
                y1 = int(y + i*tile_px)
                y2 = int(y + (i+1)*tile_px)
                patch = mask[y1:y2, x1:x2]
                row.append(1 if patch.mean() > OCC_THRESH else 0)
            grid.append(row)

        grids.append(grid)
        boxes.append((x,y,w,h))

    # sort left→right by box-x
    order = sorted(range(len(boxes)), key=lambda i: boxes[i][0])
    grids = [grids[i] for i in order]

    return grids, prev, mask, stats

def visualize(prev, mask, stats):
    # show preview & mask
    cv2.imshow("Preview", prev)
    cv2.imshow("Mask", mask*255)

    # overlay each component
    vis = cv2.cvtColor(prev, cv2.COLOR_BGR2RGB)
    for lbl in range(1, stats.shape[0]):
        x,y,w,h = stats[lbl,cv2.CC_STAT_LEFT], stats[lbl,cv2.CC_STAT_TOP], \
                  stats[lbl,cv2.CC_STAT_WIDTH], stats[lbl,cv2.CC_STAT_HEIGHT]
        area = stats[lbl,cv2.CC_STAT_AREA]
        if area < MIN_AREA_FRAC*( (bs/8.0)**2 ): continue
        cv2.rectangle(vis, (x,y), (x+w,y+h), (255,0,0), 2)

    cv2.imshow("Pieces", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__=="__main__":
    full = cv2.imread(CAPTURE_PATH)
    ui_bg   = sample_color(os.path.join(ASSETS_DIR,"inverted_background.png"))
    grid_bg = sample_color(os.path.join(ASSETS_DIR,"inverted_empty_tile.png"))

    ui_roi = find_window(full, ui_bg)
    win    = full[ui_roi[1]:ui_roi[1]+ui_roi[3],
                  ui_roi[0]:ui_roi[0]+ui_roi[2]]
    grid_roi = find_grid_bounds(win, grid_bg)
    board_roi = (ui_roi[0]+grid_roi[0],
                 ui_roi[1]+grid_roi[1],
                 grid_roi[2], grid_roi[3])

    grids, prev, mask, stats = detect_next_blocks(full, ui_roi, board_roi)

    if DEBUG and prev is not None:
        visualize(prev, mask, stats)

    # save only the grids
    with open(OUTPUT_JSON,"w") as f:
        json.dump(grids, f, indent=2)
    print(json.dumps(grids, indent=2))