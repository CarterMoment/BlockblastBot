#!/usr/bin/env python3
import cv2, numpy as np, json, os

# ——— CONFIGURATION ———
ASSETS_DIR         = "assets"
SCREENSHOT_PATH    = os.path.join(ASSETS_DIR, "latest_capture.png")
UI_BG_SAMPLE_PATH  = os.path.join(ASSETS_DIR, "inverted_background.png")
GRID_BG_SAMPLE_PATH= os.path.join(ASSETS_DIR, "inverted_empty_tile.png")
OUTPUT_JSON        = "board_matrix.json"

GRID_SIZE          = 8

# tolerances
UI_BG_TOL          = 30     # for full window detection
GRID_BG_TOL        = 20     # for grid area detection
MORPH_UI           = 25     # closing kernel for window mask
MORPH_GRID         = 7      # closing kernel for grid mask

# cell sampling
PATCH_SCALE        = 0.6
OCCUPANCY_THRESH   = 0.10   # >10% non‑bg pixels → filled

# grid contour filters
MIN_COVER_FRAC     = 0.6    # min fraction of window size the grid must span
MAX_ASPECT_DIST    = 0.25   # |w/h - 1| < 0.25 (near square)
TOP_SKIP_FRAC      = 0.10   # skip anything starting in top 10% of window


def sample_color_bgr(path):
    img = cv2.imread(path)
    h,w,_ = img.shape
    return img[h//2, w//2].astype(np.int32)


def find_window_roi(img, bg_color):
    diff   = np.linalg.norm(img.astype(np.int32)-bg_color[None,None,:],axis=2)
    mask   = (diff<UI_BG_TOL).astype(np.uint8)*255
    mask   = cv2.bitwise_not(mask)
    k      = cv2.getStructuringElement(cv2.MORPH_RECT,(MORPH_UI,MORPH_UI))
    mask   = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,k)
    cnts,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    x,y,w,h= cv2.boundingRect(max(cnts,key=lambda c:cv2.contourArea(c)))
    return x,y,w,h


def find_board_roi(win_img, grid_bg):
    H,W,_ = win_img.shape
    # mask out grid background
    diff    = np.linalg.norm(win_img.astype(np.int32)-grid_bg[None,None,:],axis=2)
    mask_bg = (diff<GRID_BG_TOL).astype(np.uint8)*255
    mask    = cv2.bitwise_not(mask_bg)
    k       = cv2.getStructuringElement(cv2.MORPH_RECT,(MORPH_GRID,MORPH_GRID))
    mask    = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,k)

    cnts,_  = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    candidates=[]
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        # skip any that start too high (UI number)
        if y < H*TOP_SKIP_FRAC: continue
        # must cover enough of the window
        if w < W*MIN_COVER_FRAC or h < H*MIN_COVER_FRAC: continue
        # must be roughly square
        if abs((w/h) - 1) > MAX_ASPECT_DIST: continue
        candidates.append((x,y,w,h))
    if candidates:
        # pick the largest area among candidates
        bx,by,bw,bh = max(candidates, key=lambda r: r[2]*r[3])
    else:
        # fallback to largest contour
        bx,by,bw,bh = max([cv2.boundingRect(c) for c in cnts], key=lambda r: r[2]*r[3])

    # square‑ize the chosen rect
    side = min(bw,bh)
    # center it vertically/horizontally within that rect
    bx += (bw-side)//2
    by += (bh-side)//2
    return bx, by, side, side


def extract_and_viz(full, ui_roi, board_roi, grid_bg):
    x0,y0,w0,h0 = ui_roi
    bx,by,bs,_  = board_roi

    viz = full.copy()
    cv2.rectangle(viz,(x0,y0),(x0+w0,y0+h0),(255,0,0),2)           # window = blue
    cv2.rectangle(viz,(x0+bx,y0+by),(x0+bx+bs,y0+by+bs),(0,255,255),2)  # board = yellow

    cell = bs//GRID_SIZE
    patch= int(cell*PATCH_SCALE)
    off  = (cell-patch)//2

    matrix=[]
    for i in range(GRID_SIZE):
        row=[]
        for j in range(GRID_SIZE):
            px = x0+bx + j*cell + off
            py = y0+by + i*cell + off
            tile = full[py:py+patch, px:px+patch].astype(np.int32)
            dist = np.linalg.norm(tile-grid_bg[None,None,:],axis=2)
            frac = np.mean(dist>GRID_BG_TOL)
            filled = 1 if frac>OCCUPANCY_THRESH else 0
            row.append(filled)
            color = (0,0,255) if filled else (0,255,0)
            cv2.rectangle(viz,
                (x0+bx + j*cell,   y0+by + i*cell),
                (x0+bx + (j+1)*cell,y0+by + (i+1)*cell),
                color,2)
        matrix.append(row)

    return matrix, viz


if __name__=="__main__":
    full     = cv2.imread(SCREENSHOT_PATH)
    ui_bg    = sample_color_bgr(UI_BG_SAMPLE_PATH)
    grid_bg  = sample_color_bgr(GRID_BG_SAMPLE_PATH)

    ui_roi   = find_window_roi(full, ui_bg)
    x0,y0,w0,h0 = ui_roi
    win_img  = full[y0:y0+h0, x0:x0+w0]

    board_roi= find_board_roi(win_img, grid_bg)
    matrix, viz = extract_and_viz(full, ui_roi, board_roi, grid_bg)

    with open(OUTPUT_JSON,"w") as f:
        json.dump(matrix,f,indent=2)
    print("✅ Saved board matrix to", OUTPUT_JSON)
    for row in matrix: print(row)

    cv2.imshow("Board Detection",viz)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
