#!/usr/bin/env python3
import cv2, numpy as np, json, os

# ——— CONFIGURATION ———
ASSETS_DIR        = "assets"
SCREENSHOT_PATH   = os.path.join(ASSETS_DIR, "latest_capture.png")
UI_BG_SAMPLE_PATH = os.path.join(ASSETS_DIR, "inverted_background.png")
GRID_BG_SAMPLE_PATH = os.path.join(ASSETS_DIR, "inverted_empty_tile.png")
OUTPUT_JSON       = "board_matrix.json"

GRID_SIZE         = 8

# tolerances
UI_BG_TOL         = 30    # window background match
GRID_BG_TOL       = 20    # grid background (empty cell) match
MORPH_UI          = 25    # morph close for window
MORPH_GRID        = 7     # morph close for grid

# where to look for grid within the window
TOP_SEARCH_FRAC   = 0.15  # start 15% down from top
BOT_SEARCH_FRAC   = 0.90  # end at 90% of height

# per‑cell sampling
PATCH_SCALE       = 0.6
OCCUPANCY_THRESH  = 0.10  # 10% non-bg pixels → filled

def sample_color(path):
    img = cv2.imread(path)
    h,w,_ = img.shape
    return img[h//2, w//2].astype(np.int32)

def find_window_roi(img, bg_color):
    diff   = np.linalg.norm(img.astype(np.int32) - bg_color[None,None,:], axis=2)
    mask   = (diff < UI_BG_TOL).astype(np.uint8)*255
    mask   = cv2.bitwise_not(mask)
    k      = cv2.getStructuringElement(cv2.MORPH_RECT, (MORPH_UI, MORPH_UI))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x,y,w,h= cv2.boundingRect(max(cnts, key=cv2.contourArea))
    return x,y,w,h

def find_board_roi(win_img, grid_bg):
    H,W,_ = win_img.shape
    y0 = int(H * TOP_SEARCH_FRAC)
    y1 = int(H * BOT_SEARCH_FRAC)
    crop = win_img[y0:y1, :]
    diff   = np.linalg.norm(crop.astype(np.int32) - grid_bg[None,None,:], axis=2)
    mask   = (diff < GRID_BG_TOL).astype(np.uint8)*255
    mask   = cv2.bitwise_not(mask)
    k      = cv2.getStructuringElement(cv2.MORPH_RECT, (MORPH_GRID, MORPH_GRID))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x,y,w,h= cv2.boundingRect(max(cnts, key=cv2.contourArea))
    side   = min(w, h)
    # map back into window coords
    return x, y + y0, side, side

def extract_and_visualize(full, ui_roi, board_roi, grid_bg):
    x0,y0,w0,h0 = ui_roi
    bx,by,bs,_  = board_roi
    viz = full.copy()
    cv2.rectangle(viz,(x0,y0),(x0+w0,y0+h0),(255,0,0),2)          # window
    cv2.rectangle(viz,(x0+bx,y0+by),(x0+bx+bs,y0+by+bs),(0,255,255),2)  # board

    cell = bs // GRID_SIZE
    patch= int(cell * PATCH_SCALE)
    off  = (cell - patch)//2

    matrix = []
    for i in range(GRID_SIZE):
        row=[]
        for j in range(GRID_SIZE):
            px = x0+bx + j*cell + off
            py = y0+by + i*cell + off
            tile = full[py:py+patch, px:px+patch].astype(np.int32)
            dist = np.linalg.norm(tile - grid_bg[None,None,:], axis=2)
            frac = np.mean(dist > GRID_BG_TOL)
            filled = 1 if frac > OCCUPANCY_THRESH else 0
            row.append(filled)
            color = (0,0,255) if filled else (0,255,0)
            cv2.rectangle(viz,
                (x0+bx + j*cell,    y0+by + i*cell),
                (x0+bx + (j+1)*cell, y0+by + (i+1)*cell),
                color, 2)
        matrix.append(row)
    return matrix, viz

if __name__=="__main__":
    full    = cv2.imread(SCREENSHOT_PATH)
    ui_bg   = sample_color(UI_BG_SAMPLE_PATH)
    grid_bg = sample_color(GRID_BG_SAMPLE_PATH)

    ui_roi    = find_window_roi(full, ui_bg)
    x0,y0,w0,h0 = ui_roi
    win_img   = full[y0:y0+h0, x0:x0+w0]

    board_roi = find_board_roi(win_img, grid_bg)
    matrix, viz = extract_and_visualize(full, ui_roi, board_roi, grid_bg)

    with open(OUTPUT_JSON,"w") as f:
        json.dump(matrix, f, indent=2)
    print(f"✅ Saved board matrix to {OUTPUT_JSON}")
    for row in matrix: print(row)

    cv2.imshow("Board Detection", viz)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
