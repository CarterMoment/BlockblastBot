#!/usr/bin/env python3
import cv2, numpy as np, json, os

# ——— CONFIG ———
ASSETS       = "assets"
SHOT         = os.path.join(ASSETS, "latest_capture.png")
UI_BG_SAMPLE = os.path.join(ASSETS, "inverted_background.png")
GRID_SAMPLE  = os.path.join(ASSETS, "inverted_empty_tile.png")
OUT_JSON     = "board_matrix.json"

GRID_SIZE    = 8

# color tolerances
UI_BG_TOL   = 30    # match UI background
GRID_BG_TOL = 20    # match grid background (empty cell)

# default crop fractions (fallback)
CROP_TOP_FRAC    = 0.15
CROP_BOTTOM_FRAC = 0.10
CROP_LEFT_FRAC   = 0.05
CROP_RIGHT_FRAC  = 0.05

PATCH_SCALE  = 0.6
OCC_THRESH   = 0.10  # >10% non‑bg pixels → filled


def sample_color(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    h, w = img.shape[:2]
    return img[h//2, w//2].astype(np.int32)


def find_window(full, ui_bg):
    diff = np.linalg.norm(full.astype(np.int32)-ui_bg[None,None,:], axis=2)
    mask = (diff < UI_BG_TOL).astype(np.uint8)*255
    mask = cv2.bitwise_not(mask)
    kern = cv2.getStructuringElement(cv2.MORPH_RECT,(25,25))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kern)
    cnts,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    x,y,w,h = cv2.boundingRect(max(cnts, key=lambda c:cv2.contourArea(c)))
    return x,y,w,h


def find_board(full, ui_roi, grid_bg):
    x0,y0,w0,h0 = ui_roi
    win = full[y0:y0+h0, x0:x0+w0]

    # 1) Detect any bright (white) region at top 25% of window
    gray = cv2.cvtColor(win, cv2.COLOR_BGR2GRAY)
    _, bright = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)
    bright = cv2.morphologyEx(bright,
                              cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_RECT,(5,5)))
    cnts,_ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    top_skip = 0
    for c in cnts:
        bx,by,bw,bh = cv2.boundingRect(c)
        # ignore tiny specs; only consider wide white overlays in top quarter
        if by < h0 * 0.25 and bw > w0 * 0.2:
            top_skip = max(top_skip, by + bh)
    # fallback to fraction if nothing found
    if top_skip == 0:
        top_skip = int(h0 * CROP_TOP_FRAC)
    else:
        top_skip += 5  # small margin

    # 2) Compute crop box excluding UI and previews
    cx1 = x0 + int(w0 * CROP_LEFT_FRAC)
    cx2 = x0 + int(w0 * (1 - CROP_RIGHT_FRAC))
    cy1 = y0 + top_skip
    cy2 = y0 + int(h0 * (1 - CROP_BOTTOM_FRAC))
    crop = full[cy1:cy2, cx1:cx2]

    # 3) Mask grid background inside that crop
    diff = np.linalg.norm(crop.astype(np.int32)-grid_bg[None,None,:], axis=2)
    mask = (diff < GRID_BG_TOL).astype(np.uint8)*255
    mask = cv2.bitwise_not(mask)
    mask = cv2.morphologyEx(mask,
                            cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_RECT,(7,7)))

    cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bx,by,bw,bh = cv2.boundingRect(max(cnts, key=lambda c:cv2.contourArea(c)))
    side = min(bw,bh)

    # map back to full coords
    return cx1+bx, cy1+by, side, side


def extract_matrix(full, board_roi, grid_bg):
    bx,by,bs,_ = board_roi
    cell_w = bs / GRID_SIZE
    cell_h = bs / GRID_SIZE
    pw = int(cell_w * PATCH_SCALE)
    ph = int(cell_h * PATCH_SCALE)

    mat = []
    for i in range(GRID_SIZE):
        row = []
        for j in range(GRID_SIZE):
            cx = int(bx + j*cell_w + cell_w/2)
            cy = int(by + i*cell_h + cell_h/2)
            patch = full[cy-ph//2:cy+ph//2, cx-pw//2:cx+pw//2].astype(np.int32)
            dist  = np.linalg.norm(patch - grid_bg[None,None,:], axis=2)
            frac  = np.mean(dist > GRID_BG_TOL)
            row.append(1 if frac > OCC_THRESH else 0)
        mat.append(row)
    return mat


def visualize(full, ui_roi, board_roi, matrix):
    vis = full.copy()
    x0,y0,w0,h0 = ui_roi
    bx,by,bs,_  = board_roi

    # window = blue, board = yellow
    cv2.rectangle(vis,(x0,y0),(x0+w0,y0+h0),(255,0,0),2)
    cv2.rectangle(vis,(bx,by),(bx+bs,by+bs),(0,255,255),2)

    cell = bs // GRID_SIZE
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            color = (0,0,255) if matrix[i][j] else (0,255,0)
            x1, y1 = bx + j*cell, by + i*cell
            cv2.rectangle(vis,(x1,y1),(x1+cell,y1+cell),color,2)

    cv2.imshow("Board Detection", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    full    = cv2.imread(SHOT)
    if full is None:
        raise FileNotFoundError(f"Cannot load {SHOT}")
    ui_bg   = sample_color(UI_BG_SAMPLE)
    grid_bg = sample_color(GRID_SAMPLE)

    ui_roi    = find_window(full, ui_bg)
    board_roi = find_board(full, ui_roi, grid_bg)
    matrix    = extract_matrix(full, board_roi, grid_bg)

    with open(OUT_JSON, "w") as f:
        json.dump(matrix, f, indent=2)
    print("✅ Saved board_matrix.json")
    for row in matrix:
        print(row)

    visualize(full, ui_roi, board_roi, matrix)


if __name__=="__main__":
    main()
