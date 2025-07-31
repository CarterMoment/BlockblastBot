#!/usr/bin/env python3
import cv2, numpy as np, json, os, math
from detect_board import sample_color, find_window, find_grid_bounds

# ——— CONFIGURATION ———
ASSETS          = "assets"
SHOT            = "latest_capture.png"
UI_BG_SAMPLE    = os.path.join(ASSETS, "inverted_background.png")
GRID_BG_SAMPLE  = os.path.join(ASSETS, "inverted_empty_tile.png")
OUTPUT_JSON     = "next_blocks.json"

# preview‑strip geometry
PREVIEW_OFFSET  = 5      # px above board bottom
STRIP_TILES     = 3.8     # how many tile‑heights to capture

# background‑vs‑block test
BG_TOL          = 40      # Euclidean BGR distance

DEBUG           = False   # set False to disable ui popups

def find_preview_strip(full, ui_roi, grid_roi):
    x0,y0,w0,h0 = ui_roi
    gx,gy,gs,_  = grid_roi
    tile_h       = gs/8.0

    top    = int(y0 + gy + gs + PREVIEW_OFFSET)
    height = int(tile_h * STRIP_TILES)
    strip  = full[top:top+height, x0:x0+w0]
    return strip, top, x0

def find_piece_boxes(preview, bg_color):
    diff = np.linalg.norm(preview.astype(np.int32) - bg_color[None,None,:], axis=2)
    mask = (diff > BG_TOL).astype(np.uint8)*255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_RECT,(5,5)))
    cnts,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        if w*h < 100:   # skip tiny specks
            continue
        boxes.append((x,y,w,h))
    boxes.sort(key=lambda b: b[0])
    return boxes, mask

def extract_matrix_from_box(preview, box, bg_color, tile_px):
    x,y,w,h = box

    # DEBUG print of raw ratios
    ratio_h = h / tile_px
    ratio_w = w / tile_px
    print(f"DEBUG box={box} tile_px={tile_px:.2f} → h/tile_px={ratio_h:.2f}, w/tile_px={ratio_w:.2f}")

    # pad by 5% of tile_px to recover any clipped edges
    pad_h = int(tile_px * 0.05)
    pad_w = int(tile_px * 0.05)
    h_eff = h + pad_h
    w_eff = w + pad_w

    # infer rows/cols with ceil
    rows = max(1, min(5, math.ceil(2 * (h / tile_px))))
    cols = max(1, min(5, math.ceil(2 * (w / tile_px))))

    # build occupancy grid by sampling cell centers
    mat = []
    for i in range(rows):
        row = []
        for j in range(cols):
            cx = int(x + (j + 0.5) * (w/cols))
            cy = int(y + (i + 0.5) * (h/rows))
            bgr = preview[cy, cx].astype(np.int32)
            dist = np.linalg.norm(bgr - bg_color)
            row.append(0 if dist < BG_TOL else 1)
        mat.append(row)

    return mat

def visualize(full, ui_roi, grid_roi, preview, mask, top, left, pieces):
    vis = full.copy()

    cv2.imshow("Preview", preview)
    cv2.imshow("Mask", mask)

    strip_top = int(ui_roi[1] + grid_roi[1] + grid_roi[3] + PREVIEW_OFFSET)
    for p in pieces:
        x,y,w,h = p["bbox"]
        cv2.rectangle(vis,
            (left + x, strip_top + y),
            (left + x + w, strip_top + y + h),
            (0,0,255), 2)

    cv2.imshow("Outlines", vis)
    cv2.waitKey(0)
    if DEBUG:
        cv2.waitKey(0)
    else:
        cv2.waitKey(1)
    cv2.destroyAllWindows()

def main():
    full    = cv2.imread(SHOT)
    ui_bg   = sample_color(UI_BG_SAMPLE)
    grid_bg = sample_color(GRID_BG_SAMPLE)

    # 1) detect window & board
    ui_roi   = find_window(full, ui_bg)
    win      = full[ui_roi[1]:ui_roi[1]+ui_roi[3],
                    ui_roi[0]:ui_roi[0]+ui_roi[2]]
    grid_roi = find_grid_bounds(win, grid_bg)

    # 2) crop preview strip
    preview, top, left = find_preview_strip(full, ui_roi, grid_roi)

    # compute tile_px from grid size
    tile_px = grid_roi[2] / 8.0

    # 3) detect piece bounding boxes
    boxes, mask = find_piece_boxes(preview, ui_bg)

    # 4) extract matrices with fudge
    pieces = []
    for box in boxes:
        mat = extract_matrix_from_box(preview, box, ui_bg, tile_px)
        pieces.append({"bbox":[*box], "matrix": mat})

    # 5) save JSON
    with open(OUTPUT_JSON, "w") as f:
        json.dump(pieces, f, indent=2)
    print("Detected pieces:", pieces)

    # 6) debug visualize
    if DEBUG:
        visualize(full, ui_roi, grid_roi, preview, mask, top, left, pieces)

if __name__=="__main__":
    main()

