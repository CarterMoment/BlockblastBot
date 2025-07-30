#!/usr/bin/env python3
import cv2, numpy as np, json, os
from detect_board import sample_color, find_window, find_grid_bounds

# ——— CONFIGURATION ———
ASSETS          = "assets"
SHOT            = "latest_capture.png"
UI_BG_SAMPLE    = os.path.join(ASSETS, "inverted_background.png")
GRID_BG_SAMPLE  = os.path.join(ASSETS, "inverted_empty_tile.png")
OUTPUT_JSON     = "next_blocks.json"

# preview‑strip geometry
PREVIEW_OFFSET  = -5      # px above board bottom
STRIP_TILES     = 2.4     # how many tile‑heights to capture

# background‑vs‑block test
BG_TOL          = 40      # Euclidean BGR distance

DEBUG           = True    # set False to disable ui popups

def find_preview_strip(full, ui_roi, grid_roi):
    x0,y0,w0,h0 = ui_roi
    gx,gy,gs,_  = grid_roi
    tile_h       = gs/8.0

    top    = int(y0 + gy + gs + PREVIEW_OFFSET)
    height = int(tile_h * STRIP_TILES)
    return full[top:top+height, x0:x0+w0], top, x0

def find_piece_boxes(preview, bg_color):
    # mask out the background color in BGR space
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
    # sort left→right
    boxes.sort(key=lambda b: b[0])
    return boxes, mask

def infer_grid_dim(w,h):
    # choose smallest rows*cols>=4 using aspect ratio
    ratio = w/float(h)
    best = None
    best_area = 1e9
    for rows in range(1,5):
        cols = max(1, int(round(ratio*rows)))
        cells = rows*cols
        if cells < 4: 
            continue
        if cells < best_area:
            best_area = cells
            best = (rows, cols)
    return best if best else (1,4)

def extract_matrix_from_box(preview, box, bg_color):
    x,y,w,h = box
    rows,cols = infer_grid_dim(w,h)
    cell_w = w/cols
    cell_h = h/rows

    mat = []
    for i in range(rows):
        row = []
        for j in range(cols):
            cx = int(x + (j+0.5)*cell_w)
            cy = int(y + (i+0.5)*cell_h)
            bgr = preview[cy, cx].astype(np.int32)
            dist = np.linalg.norm(bgr - bg_color)
            row.append(0 if dist < BG_TOL else 1)
        mat.append(row)
    return mat

def visualize(full, ui_roi, grid_roi, preview, mask, top, left, pieces):
    vis = full.copy()
    # draw preview strip and mask
    cv2.imshow("Preview", preview)
    cv2.imshow("Mask", mask)

    # outline each piece
    for p in pieces:
        x,y,w,h = p["bbox"]
        cv2.rectangle(vis,
            (left + x, top + y),
            (left + x + w, top + y + h),
            (0,0,255), 2)

    cv2.imshow("Outlines", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    full    = cv2.imread(SHOT)
    ui_bg   = sample_color(UI_BG_SAMPLE)
    grid_bg = sample_color(GRID_BG_SAMPLE)

    # 1) find window & board
    ui_roi   = find_window(full, ui_bg)
    win      = full[ui_roi[1]:ui_roi[1]+ui_roi[3],
                    ui_roi[0]:ui_roi[0]+ui_roi[2]]
    grid_roi = find_grid_bounds(win, grid_bg)

    # 2) crop preview strip & get bg color
    preview, top, left = find_preview_strip(full, ui_roi, grid_roi)
    bg_color = ui_bg  # your inverted_background sample is BGR

    # 3) detect piece bboxes
    boxes, mask = find_piece_boxes(preview, bg_color)

    # 4) turn each into a matrix
    pieces = []
    for box in boxes:
        mat = extract_matrix_from_box(preview, box, bg_color)
        pieces.append({"bbox":[*box], "matrix": mat})

    # 5) save
    with open(OUTPUT_JSON, "w") as f:
        json.dump(pieces, f, indent=2)
    print("Detected pieces:", pieces)

    # 6) debug
    if DEBUG:
        visualize(full, ui_roi, grid_roi, preview, mask, top, left, pieces)

if __name__=="__main__":
    main()
