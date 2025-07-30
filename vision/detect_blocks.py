#!/usr/bin/env python3
import cv2, numpy as np, json, os
from detect_board import sample_color, find_window, find_grid_bounds

# ——— CONFIGURATION ———
ASSETS           = "assets"
SHOT             = "latest_capture.png"
UI_BG_SAMPLE     = os.path.join(ASSETS, "inverted_background.png")
GRID_BG_SAMPLE   = os.path.join(ASSETS, "inverted_empty_tile.png")
OUTPUT_JSON      = "next_blocks.json"

# preview strip params
PREVIEW_OFFSET   = -5     # px above board bottom
STRIP_TILES      = 2.4    # how many tile‑heights to grab

# HSV tolerances
HUE_TOL          = 10
SAT_TOL          = 60
VAL_TOL          = 60

DEBUG            = True   # show debug windows

def detect_blocks(full_img, ui_roi, grid_roi):
    """
    Returns list of dicts, one per detected piece:
      {
        "bbox": [x,y,w,h],        # relative to the preview strip
        "matrix": [[0,1..], ...]  # rows×cols 0/1 occupancy
      }
    plus the raw preview crop and mask for optional viz.
    """
    x0,y0,w0,h0 = ui_roi
    gx,gy,gs,_  = grid_roi

    # 1) crop just under the board
    tile_h   = gs/8.0
    top      = int(y0 + gy + gs + PREVIEW_OFFSET)
    strip_h  = int(tile_h * STRIP_TILES)
    preview  = full_img[top:top+strip_h, x0:x0+w0]
    if preview.size == 0:
        return [], preview, None

    # 2) HSV‑mask out grid background
    grid_bgr  = cv2.imread(GRID_BG_SAMPLE)
    gh,gw     = grid_bgr.shape[:2]
    bg_px     = grid_bgr[gh//2, gw//2]
    bg_hsv    = cv2.cvtColor(bg_px[None,None,:], cv2.COLOR_BGR2HSV)[0,0]
    hsv       = cv2.cvtColor(preview, cv2.COLOR_BGR2HSV)
    lower     = np.array([bg_hsv[0]-HUE_TOL, bg_hsv[1]-SAT_TOL, bg_hsv[2]-VAL_TOL])
    upper     = np.array([bg_hsv[0]+HUE_TOL, bg_hsv[1]+SAT_TOL, bg_hsv[2]+VAL_TOL])
    lower     = np.clip(lower, [0,0,0], [179,255,255]).astype(np.uint8)
    upper     = np.clip(upper, [0,0,0], [179,255,255]).astype(np.uint8)
    bg_mask   = cv2.inRange(hsv, lower, upper)
    piece_mask= cv2.bitwise_not(bg_mask)
    piece_mask= cv2.medianBlur(piece_mask, 5)

    # 3) find each blob
    n, labels, stats, _ = cv2.connectedComponentsWithStats(piece_mask, 8)
    tile_px  = gs/8.0
    min_area = 0.2 * tile_px * tile_px
    pieces   = []

    for lbl in range(1, n):
        area = stats[lbl, cv2.CC_STAT_AREA]
        if area < min_area:
            continue

        x,y,w,h = (stats[lbl, cv2.CC_STAT_LEFT],
                   stats[lbl, cv2.CC_STAT_TOP],
                   stats[lbl, cv2.CC_STAT_WIDTH],
                   stats[lbl, cv2.CC_STAT_HEIGHT])
        blob = piece_mask[y:y+h, x:x+w]

        # 4) infer rows/cols from area
        avg_sq = np.sqrt(area/4.0)
        rows   = max(1, min(4, int(round(h/avg_sq))))
        cols   = max(1, min(4, int(round(w/avg_sq))))

        # 5) build occupancy matrix
        matrix = []
        for i in range(rows):
            row = []
            for j in range(cols):
                x1 = int(j * w/cols)
                y1 = int(i * h/rows)
                x2 = int((j+1) * w/cols)
                y2 = int((i+1) * h/rows)
                cell = blob[y1:y2, x1:x2]
                row.append(1 if cell.mean() > 0.3 else 0)
            matrix.append(row)

        pieces.append({
            "bbox": [int(x), int(y), int(w), int(h)],
            "matrix": matrix
        })

    # 6) sort left→right
    pieces.sort(key=lambda p: p["bbox"][0])
    return pieces, preview, piece_mask

def visualize(full, ui_roi, grid_roi, preview, mask, pieces):
    vis = full.copy()
    x0,y0,_,_ = ui_roi
    gx,gy,gs,_= grid_roi

    # outline each piece on full image
    strip_top = int(y0 + gy + gs + PREVIEW_OFFSET)
    for p in pieces:
        x,y,w,h = p["bbox"]
        cv2.rectangle(vis,
            (x0+x, strip_top+y),
            (x0+x+w, strip_top+y+h),
            (0,0,255), 2)

    cv2.imshow("Preview", preview)
    cv2.imshow("Mask", mask*255)
    cv2.imshow("Outlines", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    full      = cv2.imread(SHOT)
    ui_bg     = sample_color(UI_BG_SAMPLE)
    grid_bg   = sample_color(GRID_BG_SAMPLE)

    ui_roi    = find_window(full, ui_bg)
    win       = full[ui_roi[1]:ui_roi[1]+ui_roi[3],
                     ui_roi[0]:ui_roi[0]+ui_roi[2]]
    grid_roi  = find_grid_bounds(win, grid_bg)

    pieces, preview, mask = detect_blocks(full, ui_roi, grid_roi)

    # save as an array of objects
    with open(OUTPUT_JSON, "w") as f:
        json.dump(pieces, f, indent=2)
    print("Detected pieces:", pieces)

    if DEBUG:
        visualize(full, ui_roi, grid_roi, preview, mask, pieces)

if __name__=="__main__":
    main()
