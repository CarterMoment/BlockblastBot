#!/usr/bin/env python3
import cv2, numpy as np, json, os

# ——— CONFIGURATION ———
ASSETS        = "assets"
SHOT          = os.path.join(ASSETS, "latest_capture.png")
UI_BG_SAMPLE  = os.path.join(ASSETS, "inverted_background.png")
GRID_SAMPLE   = os.path.join(ASSETS, "inverted_empty_tile.png")
OUT_JSON      = "board_matrix.json"

GRID_SIZE     = 8

UI_BG_TOL       = 30     # tolerance for UI background detection
GRID_BG_TOL     = 20     # tolerance for grid background detection
ROW_COL_FRAC    = 0.60   # require 60% of pixels to match grid‑bg for a row/col
PATCH_SCALE     = 0.6    # 60% central patch for occupancy test
OCC_THRESH      = 0.10   # >10% non‑bg pixels → filled

def sample_color(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not load sample: {path}")
    h, w = img.shape[:2]
    return img[h//2, w//2].astype(np.int32)

def find_window(full, ui_bg):
    diff = np.linalg.norm(full.astype(np.int32) - ui_bg[None,None,:], axis=2)
    mask = (diff < UI_BG_TOL).astype(np.uint8) * 255
    mask = cv2.bitwise_not(mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25,25))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    cnts,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    x,y,w,h = cv2.boundingRect(max(cnts, key=lambda c:cv2.contourArea(c)))
    return x,y,w,h

def find_grid_bounds(win, grid_bg):
    h, w = win.shape[:2]
    # build a binary mask of grid-background pixels
    diff = np.linalg.norm(win.astype(np.int32) - grid_bg[None,None,:], axis=2)
    mask = (diff < GRID_BG_TOL).astype(np.uint8)

    # Scan top→down for first grid row
    top = 0
    for i in range(h):
        if mask[i,:].mean() >= ROW_COL_FRAC:
            top = i
            break
    # Scan bottom→up for last grid row
    bottom = h-1
    for i in range(h-1, -1, -1):
        if mask[i,:].mean() >= ROW_COL_FRAC:
            bottom = i
            break
    # Scan left→right for first grid column
    left = 0
    for j in range(w):
        if mask[:,j].mean() >= ROW_COL_FRAC:
            left = j
            break
    # Scan right→left for last grid column
    right = w-1
    for j in range(w-1, -1, -1):
        if mask[:,j].mean() >= ROW_COL_FRAC:
            right = j
            break

    # Force a square and center it
    width  = right - left + 1
    height = bottom - top + 1
    side   = min(width, height)
    dx = (width - side) // 2
    dy = (height - side) // 2

    return left + dx, top + dy, side, side

def extract_matrix(full, roi, grid_bg):
    x, y, side, _ = roi
    board = full[y:y+side, x:x+side]
    cell_w = side / GRID_SIZE
    cell_h = side / GRID_SIZE
    pw = int(cell_w * PATCH_SCALE)
    ph = int(cell_h * PATCH_SCALE)

    matrix = []
    for i in range(GRID_SIZE):
        row = []
        for j in range(GRID_SIZE):
            cx = int(j*cell_w + cell_w/2)
            cy = int(i*cell_h + cell_h/2)
            patch = board[cy-ph//2:cy+ph//2, cx-pw//2:cx+pw//2].astype(np.int32)
            dist = np.linalg.norm(patch - grid_bg[None,None,:], axis=2)
            frac = np.mean(dist > GRID_BG_TOL)
            row.append(1 if frac > OCC_THRESH else 0)
        matrix.append(row)
    return matrix

def visualize(full, ui_roi, grid_roi, matrix):
    vis = full.copy()
    x0,y0,w0,h0 = ui_roi
    gx,gy,gs,_ = grid_roi
    # draw window ROI (blue) and grid ROI (yellow)
    cv2.rectangle(vis, (x0,y0), (x0+w0,y0+h0), (255,0,0), 2)
    cv2.rectangle(vis, (x0+gx,y0+gy), (x0+gx+gs,y0+gy+gs), (0,255,255), 2)

    cell = gs // GRID_SIZE
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            color = (0,0,255) if matrix[i][j] else (0,255,0)
            x1 = x0 + gx + j*cell
            y1 = y0 + gy + i*cell
            cv2.rectangle(vis, (x1,y1), (x1+cell,y1+cell), color, 2)

    cv2.imshow("Board Detection", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    full    = cv2.imread(SHOT)
    ui_bg   = sample_color(UI_BG_SAMPLE)
    grid_bg = sample_color(GRID_SAMPLE)

    # 1. locate QuickTime window
    ui_roi    = find_window(full, ui_bg)
    x0,y0,w0,h0 = ui_roi
    win_img   = full[y0:y0+h0, x0:x0+w0]

    # 2. find grid bounds within that window
    gx,gy,gs,gd = find_grid_bounds(win_img, grid_bg)

    # 3. extract 8×8 matrix
    matrix = extract_matrix(win_img, (gx,gy,gs,gd), grid_bg)

    # 4. save JSON
    with open(OUT_JSON, "w") as f:
        json.dump(matrix, f, indent=2)
    print(f"✅ Saved board matrix to {OUT_JSON}")
    for row in matrix:
        print(row)

    # 5. visualize overlay
    visualize(full, ui_roi, (gx,gy,gs,gd), matrix)

if __name__=="__main__":
    main()
