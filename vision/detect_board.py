#!/usr/bin/env python3
import cv2, numpy as np, json, os

# ——— CONFIGURATION ———
ASSETS_DIR          = "assets"
SCREENSHOT_PATH     = os.path.join(ASSETS_DIR, "latest_capture.png")
UI_BG_SAMPLE_PATH   = os.path.join(ASSETS_DIR, "inverted_background.png")
GRID_BG_SAMPLE_PATH = os.path.join(ASSETS_DIR, "inverted_empty_tile.png")
OUTPUT_JSON         = "board_matrix.json"

GRID_SIZE          = 8

# color tolerances (BGR distance)
UI_BG_TOL   = 30     # for locating the window
GRID_BG_TOL = 20     # for occupancy test inside cells

# morphological sizes
MORPH_UI    = 25     # to close holes in window mask
MORPH_GRID  = 7      # to close holes in grid mask

# how much of the window to ignore at top/bottom
TOP_CROP_FRAC   = 0.15   # ignore top 15% (score UI)
BOT_CROP_FRAC   = 0.10   # ignore bottom 10% (preview pieces)

# occupancy test
PATCH_SCALE     = 0.6    # sample central 60% of each cell
OCC_THRESH      = 0.10   # >10% non‑bg pixels → filled

def sample_bgr(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    h,w,_ = img.shape
    return img[h//2, w//2].astype(np.int32)

def find_window_roi(full, ui_bg):
    """Mask out UI background, close, take largest contour = app window."""
    diff = np.linalg.norm(full.astype(np.int32) - ui_bg[None,None,:], axis=2)
    mask = (diff < UI_BG_TOL).astype(np.uint8)*255
    mask = cv2.bitwise_not(mask)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (MORPH_UI, MORPH_UI))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x,y,w,h = cv2.boundingRect(max(cnts, key=lambda c:cv2.contourArea(c)))
    return x,y,w,h

def detect_grid_lines(board_gray):
    """Return (vertical_lines, horizontal_lines) from Hough on morphological masks."""
    H,W = board_gray.shape

    # edge map
    blur = cv2.GaussianBlur(board_gray, (5,5),0)
    edges = cv2.Canny(blur, 50, 150)

    # vertical mask → HoughLinesP for near‑vertical
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, H//10))
    v_mask = cv2.morphologyEx(edges, cv2.MORPH_OPEN, v_kernel)
    v_lines = cv2.HoughLinesP(v_mask, 1, np.pi/180,
                              threshold=H//2,
                              minLineLength=H*0.7,
                              maxLineGap=20) or []

    # horizontal mask → HoughLinesP for near‑horizontal
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (W//10, 1))
    h_mask = cv2.morphologyEx(edges, cv2.MORPH_OPEN, h_kernel)
    h_lines = cv2.HoughLinesP(h_mask, 1, np.pi/180,
                              threshold=W//2,
                              minLineLength=W*0.7,
                              maxLineGap=20) or []

    # filter & average coords
    verts = [((x1,y1),(x2,y2)) for x1,y1,x2,y2 in v_lines.reshape(-1,4)]
    hors  = [((x1,y1),(x2,y2)) for x1,y1,x2,y2 in h_lines.reshape(-1,4)]
    return verts, hors

def cluster_positions(pos_list, max_dist):
    """Agglomerative cluster of scalars: average within max_dist."""
    if not pos_list: return []
    sorted_p = sorted(pos_list)
    clusters = [[sorted_p[0]]]
    for p in sorted_p[1:]:
        if abs(p - np.mean(clusters[-1])) <= max_dist:
            clusters[-1].append(p)
        else:
            clusters.append([p])
    return [int(np.mean(c)) for c in clusters]

def find_cell_boundaries(board):
    """Detect 9 vertical & 9 horizontal boundaries, return sorted x[], y[]."""
    gray = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
    verts, hors = detect_grid_lines(gray)

    H,W = gray.shape
    # extract center positions
    xs = [ (x1+x2)//2 for (x1,y1),(x2,y2) in verts ]
    ys = [ (y1+y2)//2 for (x1,y1),(x2,y2) in hors ]

    # cluster by half‐cell spacing
    cell_w = W / GRID_SIZE
    cell_h = H / GRID_SIZE
    xs_cl = cluster_positions(xs, max_dist=cell_w/2)
    ys_cl = cluster_positions(ys, max_dist=cell_h/2)

    # ensure we have at least 9 lines: add borders if missing
    if len(xs_cl) < GRID_SIZE+1:
        xs_cl = [0] + xs_cl + [W]
    if len(ys_cl) < GRID_SIZE+1:
        ys_cl = [0] + ys_cl + [H]

    # sort & pick the central GRID_SIZE+1 lines
    xs_cl = sorted(xs_cl)
    ys_cl = sorted(ys_cl)
    # take the middle GRID_SIZE+1 if more
    if len(xs_cl) > GRID_SIZE+1:
        start = (len(xs_cl) - (GRID_SIZE+1))//2
        xs_cl = xs_cl[start:start+GRID_SIZE+1]
    if len(ys_cl) > GRID_SIZE+1:
        start = (len(ys_cl) - (GRID_SIZE+1))//2
        ys_cl = ys_cl[start:start+GRID_SIZE+1]

    return xs_cl, ys_cl

def classify_cells(board, xs, ys, grid_bg):
    """Build 8×8 matrix by occupancy test inside each cell patch."""
    mat=[]
    H,W = board.shape[:2]
    patch = int((W/GRID_SIZE) * PATCH_SCALE)
    for i in range(GRID_SIZE):
        row=[]
        for j in range(GRID_SIZE):
            x0,x1 = xs[j], xs[j+1]
            y0,y1 = ys[i], ys[i+1]
            # sample central patch
            cx = int((x0+x1)/2 - patch/2)
            cy = int((y0+y1)/2 - patch/2)
            tile = board[cy:cy+patch, cx:cx+patch].astype(np.int32)
            # occupancy = fraction of pixels != grid_bg
            dist = np.linalg.norm(tile - grid_bg[None,None,:], axis=2)
            frac = np.mean(dist > GRID_BG_TOL)
            row.append(1 if frac>OCC_THRESH else 0)
        mat.append(row)
    return mat

def main():
    full = cv2.imread(SCREENSHOT_PATH)
    ui_bg = sample_bgr(UI_BG_SAMPLE_PATH)
    grid_bg = sample_bgr(GRID_BG_SAMPLE_PATH)

    # 1) find QuickTime window
    wx,wy,ww,wh = find_window_roi(full, ui_bg)
    win = full[wy:wy+wh, wx:wx+ww]

    # 2) crop out top UI and bottom preview
    y0 = int(wh*TOP_CROP_FRAC)
    y1 = int(wh*(1 - BOT_CROP_FRAC))
    board_roi = win[y0:y1, :]
  
    # 3) detect cell boundaries
    xs, ys = find_cell_boundaries(board_roi)

    # map back coords to full image
    xs = [int(x + wx) for x in xs]
    ys = [int(y + wy + y0) for y in ys]

    # 4) extract board patch for classification
    board_full = full[ ys[0]:ys[-1], xs[0]:xs[-1] ]

    # 5) classify
    matrix = classify_cells(board_full, 
                            [x - xs[0] for x in xs], 
                            [y - ys[0] for y in ys],
                            grid_bg)

    # 6) visualize
    viz = full.copy()
    # draw lines
    for x in xs: cv2.line(viz, (x, ys[0]), (x, ys[-1]), (0,255,0),1)
    for y in ys: cv2.line(viz, (xs[0], y), (xs[-1], y), (0,255,0),1)
    # overlay occupancy
    cell_w = (xs[1]-xs[0])
    cell_h = (ys[1]-ys[0])
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            color = (0,0,255) if matrix[i][j] else (0,255,0)
            x0 = xs[j]; y0 = ys[i]
            cv2.rectangle(viz, (x0,y0), (x0+cell_w,y0+cell_h), color, 2)

    cv2.imshow("Grid & Cells", viz)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 7) save JSON
    with open(OUTPUT_JSON, "w") as f:
        json.dump(matrix, f, indent=2)
    print(f"✅ board_matrix.json saved")
    for row in matrix: print(row)

if __name__=="__main__":
    main()
