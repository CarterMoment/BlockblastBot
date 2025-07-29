#!/usr/bin/env python3
import cv2, numpy as np, json, os

# ——— CONFIGURATION ———
ASSETS       = "assets"
SHOT         = os.path.join(ASSETS, "latest_capture.png")
UI_BG_SAMPLE = os.path.join(ASSETS, "inverted_background.png")
GRID_SAMPLE  = os.path.join(ASSETS, "inverted_empty_tile.png")
OUT_JSON     = "board_matrix.json"

GRID_SIZE    = 8

# tolerances
UI_BG_TOL       = 30    # match light‐brown UI
GRID_BG_TOL     = 20    # match dark grid background

# how much to crop away before we search for the board
CROP_TOP       = 0.15   # drop top 15% (score area)
CROP_BOTTOM    = 0.10   # drop bottom 10% (preview pieces)
CROP_LEFT      = 0.05   # drop left  5% (side UI)
CROP_RIGHT     = 0.05   # drop right 5% (side UI)

# cell‑level test
PATCH_SCALE     = 0.6   # sample central 60% of each cell
OCC_THRESH      = 0.10  # >10% non‑bg pixels → filled

def sample_color(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    h,w = img.shape[:2]
    return img[h//2, w//2].astype(np.int32)

def find_window(full, ui_bg):
    diff = np.linalg.norm(full.astype(np.int32)-ui_bg[None,None,:],axis=2)
    mask = (diff<UI_BG_TOL).astype(np.uint8)*255
    mask = cv2.bitwise_not(mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(25,25))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    cnts,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    x,y,w,h = cv2.boundingRect(max(cnts,key=cv2.contourArea))
    return x,y,w,h

def find_board(full, ui_roi, grid_bg):
    x0,y0,w0,h0 = ui_roi
    # crop away UI chrome & previews
    cx1 = x0 + int(w0 * CROP_LEFT)
    cx2 = x0 + int(w0 * (1-CROP_RIGHT))
    cy1 = y0 + int(h0 * CROP_TOP)
    cy2 = y0 + int(h0 * (1-CROP_BOTTOM))
    crop = full[cy1:cy2, cx1:cx2]

    diff = np.linalg.norm(crop.astype(np.int32)-grid_bg[None,None,:],axis=2)
    mask = (diff<GRID_BG_TOL).astype(np.uint8)*255
    mask = cv2.bitwise_not(mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(7,7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    cnts,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    bx,by,bw,bh = cv2.boundingRect(max(cnts,key=cv2.contourArea))
    # make square
    side = min(bw,bh)
    # map back to full coords
    return cx1+bx, cy1+by, side, side

def extract_matrix(full, board_roi, grid_bg):
    bx,by,bs,_ = board_roi
    cell_w = bs / GRID_SIZE
    cell_h = bs / GRID_SIZE
    patch_w = int(cell_w * PATCH_SCALE)
    patch_h = int(cell_h * PATCH_SCALE)

    matrix = []
    for i in range(GRID_SIZE):
        row=[]
        for j in range(GRID_SIZE):
            # compute cell center
            cx = int(bx + j*cell_w + cell_w/2)
            cy = int(by + i*cell_h + cell_h/2)
            # extract patch
            x1 = cx - patch_w//2
            y1 = cy - patch_h//2
            patch = full[y1:y1+patch_h, x1:x1+patch_w].astype(np.int32)
            # occupancy test
            dist = np.linalg.norm(patch - grid_bg[None,None,:], axis=2)
            frac = np.mean(dist > GRID_BG_TOL)
            row.append(1 if frac > OCC_THRESH else 0)
        matrix.append(row)
    return matrix

def visualize(full, ui_roi, board_roi, matrix):
    vis = full.copy()
    x0,y0,w0,h0 = ui_roi
    bx,by,bs,_ = board_roi
    # draw window
    cv2.rectangle(vis,(x0,y0),(x0+w0,y0+h0),(255,0,0),2)
    # draw board
    cv2.rectangle(vis,(bx,by),(bx+bs,by+bs),(0,255,255),2)

    cell = int(bs/GRID_SIZE)
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            color = (0,0,255) if matrix[i][j] else (0,255,0)
            x1 = bx + j*cell
            y1 = by + i*cell
            cv2.rectangle(vis,(x1,y1),(x1+cell,y1+cell),color,2)

    cv2.imshow("Detect Board", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    full   = cv2.imread(SHOT)
    ui_bg  = sample_color(UI_BG_SAMPLE)
    grid_bg= sample_color(GRID_SAMPLE)

    ui_roi    = find_window(full, ui_bg)
    board_roi = find_board(full, ui_roi, grid_bg)
    matrix    = extract_matrix(full, board_roi, grid_bg)

    # save
    with open(OUT_JSON,"w") as f:
        json.dump(matrix, f, indent=2)
    print("✅ board_matrix.json saved")
    for row in matrix: print(row)

    # debug
    visualize(full, ui_roi, board_roi, matrix)

if __name__=="__main__":
    main()
