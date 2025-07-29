#!/usr/bin/env python3
import cv2, numpy as np, json, os

# ——— CONFIGURATION ———
ASSETS        = "assets"
SHOT          = os.path.join(ASSETS, "latest_capture.png")
UI_BG_SAMPLE  = os.path.join(ASSETS, "inverted_background.png")
GRID_SAMPLE   = os.path.join(ASSETS, "inverted_empty_tile.png")
OUT_JSON      = "board_matrix.json"

UI_BG_TOL       = 30     # tolerance to match UI background
GRID_BG_TOL     = 20     # tolerance to match grid background (empty cell)
MORPH_UI        = 25     # kernel for closing window mask

GRID_SIZE       = 8
ROW_COL_FRAC    = 0.60   # require 60% of pixels in a line to match grid-bg
PATCH_SCALE     = 0.6    # sample 60%×60% central patch per cell
OCC_THRESH      = 0.10   # >10% non‑bg → filled

def sample_color(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    h,w = img.shape[:2]
    return img[h//2, w//2].astype(np.int32)

def find_window(full, ui_bg):
    diff = np.linalg.norm(full.astype(np.int32)-ui_bg[None,None,:],axis=2)
    mask = (diff < UI_BG_TOL).astype(np.uint8)*255
    mask = cv2.bitwise_not(mask)
    k = cv2.getStructuringElement(cv2.MORPH_RECT,(MORPH_UI,MORPH_UI))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    cnts,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    x,y,w,h = cv2.boundingRect(max(cnts, key=lambda c:cv2.contourArea(c)))
    return x,y,w,h

def find_grid_bounds(win_img, grid_bg):
    h,w = win_img.shape[:2]
    # create binary mask of grid‑background pixels
    diff = np.linalg.norm(win_img.astype(np.int32)-grid_bg[None,None,:],axis=2)
    grid_mask = (diff < GRID_BG_TOL).astype(np.uint8)

    # scan top→down for first row with enough grid-bg
    row_counts = grid_mask.mean(axis=1)  # fraction per row
    top = next(i for i,v in enumerate(row_counts) if v >= ROW_COL_FRAC)
    bottom = h - next(i for i,v in enumerate(row_counts[::-1]) if v >= ROW_COL_FRAC)

    # scan left→right for first col with enough grid-bg
    col_counts = grid_mask.mean(axis=0)
    left = next(j for j,v in enumerate(col_counts) if v >= ROW_COL_FRAC)
    right = w - next(j for j,v in enumerate(col_counts[::-1]) if v >= ROW_COL_FRAC)

    # force a square
    side = min(right-left, bottom-top)
    # center square in that box
    cx = left + ( (right-left) - side )//2
    cy = top  + ( (bottom-top) - side )//2
    return cx, cy, side, side

def extract_matrix(full, roi, grid_bg):
    x,y,side,_ = roi
    board = full[y:y+side, x:x+side]
    cell = side / GRID_SIZE
    pw = int(cell * PATCH_SCALE)
    ph = int(cell * PATCH_SCALE)

    matrix=[]
    for i in range(GRID_SIZE):
        row=[]
        for j in range(GRID_SIZE):
            cx = int(j*cell + cell/2)
            cy = int(i*cell + cell/2)
            patch = board[cy-ph//2:cy+ph//2, cx-pw//2:cx+pw//2].astype(np.int32)
            dist = np.linalg.norm(patch - grid_bg[None,None,:],axis=2)
            frac = np.mean(dist > GRID_BG_TOL)
            row.append(1 if frac > OCC_THRESH else 0)
        matrix.append(row)
    return matrix, board

def visualize(full, ui_roi, grid_roi, matrix):
    vis = full.copy()
    x0,y0,w0,h0 = ui_roi
    gx,gy,gs,_ = grid_roi
    cv2.rectangle(vis,(x0,y0),(x0+w0,y0+h0),(255,0,0),2)          # window
    cv2.rectangle(vis,(x0+gx,y0+gy),(x0+gx+gs,y0+gy+gs),(0,255,255),2)  # grid

    cell = gs//GRID_SIZE
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            color=(0,0,255) if matrix[i][j] else (0,255,0)
            x1 = x0+gx + j*cell
            y1 = y0+gy + i*cell
            cv2.rectangle(vis,(x1,y1),(x1+cell,y1+cell),color,2)
    cv2.imshow("Detection",vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    full   = cv2.imread(SHOT)
    ui_bg  = sample_color(UI_BG_SAMPLE)
    grid_bg= sample_color(GRID_SAMPLE)

    ui_roi   = find_window(full, ui_bg)
    x0,y0,w0,h0 = ui_roi
    win      = full[y0:y0+h0, x0:x0+w0]

    grid_roi = find_grid_bounds(win, grid_bg)
    # map back to full coords
    gx,gy,gs,gd = grid_roi
    grid_roi_full = (x0+gx, y0+gy, gs, gs)

    matrix, _ = extract_matrix(full, grid_roi_full, grid_bg)

    with open(OUT_JSON,"w") as f:
        json.dump(matrix,f,indent=2)
    print("✅ board_matrix.json saved")
    for row in matrix: print(row)

    visualize(full, ui_roi, grid_roi_full, matrix)

if __name__=="__main__":
    main()
