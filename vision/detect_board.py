#!/usr/bin/env python3
import cv2, numpy as np, json, os

# ——— CONFIGURATION ———
ASSETS_DIR          = "assets"
SCREENSHOT_PATH     = os.path.join(ASSETS_DIR, "latest_capture.png")
UI_BG_SAMPLE_PATH   = os.path.join(ASSETS_DIR, "inverted_background.png")
GRID_BG_SAMPLE_PATH = os.path.join(ASSETS_DIR, "inverted_empty_tile.png")
OUTPUT_JSON         = "board_matrix.json"

GRID_SIZE          = 8

# color tolerances
UI_BG_TOL   = 30    # match QuickTime window
GRID_BG_TOL = 20    # match grid‑background (empty cell)

# morphology
MORPH_UI    = 25
MORPH_GRID  = 7

# crop fractions
TOP_CROP_FRAC   = 0.15
BOT_CROP_FRAC   = 0.10

# per‑cell sample
PATCH_SCALE     = 0.6
OCC_THRESH      = 0.10


def sample_bgr(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    h,w,_ = img.shape
    return img[h//2, w//2].astype(np.int32)


def find_window_roi(full, ui_bg):
    diff = np.linalg.norm(full.astype(np.int32) - ui_bg[None,None,:], axis=2)
    mask = (diff < UI_BG_TOL).astype(np.uint8)*255
    mask = cv2.bitwise_not(mask)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (MORPH_UI, MORPH_UI))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x,y,w,h = cv2.boundingRect(max(cnts, key=lambda c:cv2.contourArea(c)))
    return x,y,w,h


def detect_grid_lines(board_gray):
    H,W = board_gray.shape

    # Edge map
    blur  = cv2.GaussianBlur(board_gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)

    # Vertical
    vk   = cv2.getStructuringElement(cv2.MORPH_RECT, (1, H//10))
    vmask= cv2.morphologyEx(edges, cv2.MORPH_OPEN, vk)
    raw_v= cv2.HoughLinesP(vmask,1,np.pi/180,
                           threshold=H//2,
                           minLineLength=int(H*0.7),
                           maxLineGap=20)
    verts=[]
    if raw_v is not None:
        for x1,y1,x2,y2 in raw_v.reshape(-1,4):
            verts.append(((x1,y1),(x2,y2)))

    # Horizontal
    hk   = cv2.getStructuringElement(cv2.MORPH_RECT, (W//10, 1))
    hmask= cv2.morphologyEx(edges, cv2.MORPH_OPEN, hk)
    raw_h= cv2.HoughLinesP(hmask,1,np.pi/180,
                           threshold=W//2,
                           minLineLength=int(W*0.7),
                           maxLineGap=20)
    hors=[]
    if raw_h is not None:
        for x1,y1,x2,y2 in raw_h.reshape(-1,4):
            hors.append(((x1,y1),(x2,y2)))

    return verts, hors


def cluster_positions(pos_list, max_dist):
    if not pos_list:
        return []
    pts = sorted(pos_list)
    clusters = [[pts[0]]]
    for p in pts[1:]:
        if abs(p - np.mean(clusters[-1])) <= max_dist:
            clusters[-1].append(p)
        else:
            clusters.append([p])
    return [int(np.mean(c)) for c in clusters]


def find_cell_boundaries(board):
    gray = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
    verts, hors = detect_grid_lines(gray)
    H,W = gray.shape

    xs = [ (x1+x2)//2 for (x1,y1),(x2,y2) in verts ]
    ys = [ (y1+y2)//2 for (x1,y1),(x2,y2) in hors ]

    cell_w = W/GRID_SIZE
    cell_h = H/GRID_SIZE

    xs_cl = cluster_positions(xs, max_dist=cell_w/2)
    ys_cl = cluster_positions(ys, max_dist=cell_h/2)

    # Ensure exactly GRID_SIZE+1 boundaries
    if len(xs_cl) < GRID_SIZE+1:
        xs_cl = [0] + xs_cl + [W]
    if len(ys_cl) < GRID_SIZE+1:
        ys_cl = [0] + ys_cl + [H]

    xs_cl = sorted(xs_cl)
    ys_cl = sorted(ys_cl)
    if len(xs_cl) > GRID_SIZE+1:
        start = (len(xs_cl) - (GRID_SIZE+1))//2
        xs_cl = xs_cl[start:start+GRID_SIZE+1]
    if len(ys_cl) > GRID_SIZE+1:
        start = (len(ys_cl) - (GRID_SIZE+1))//2
        ys_cl = ys_cl[start:start+GRID_SIZE+1]

    return xs_cl, ys_cl


def classify_cells(board, xs, ys, grid_bg):
    patch = int((board.shape[1]/GRID_SIZE)*PATCH_SCALE)
    mat=[]
    for i in range(GRID_SIZE):
        row=[]
        for j in range(GRID_SIZE):
            x0,x1 = xs[j], xs[j+1]
            y0,y1 = ys[i], ys[i+1]
            cx,cy = (x0+x1)//2, (y0+y1)//2
            tile = board[cy-patch//2:cy+patch//2, cx-patch//2:cx+patch//2].astype(np.int32)
            dist = np.linalg.norm(tile - grid_bg[None,None,:], axis=2)
            frac = np.mean(dist > GRID_BG_TOL)
            row.append(1 if frac>OCC_THRESH else 0)
        mat.append(row)
    return mat


if __name__=="__main__":
    full   = cv2.imread(SCREENSHOT_PATH)
    ui_bg  = sample_bgr(UI_BG_SAMPLE_PATH)
    grid_bg= sample_bgr(GRID_BG_SAMPLE_PATH)

    # 1) Window ROI
    x,y,w,h = find_window_roi(full, ui_bg)
    win = full[y:y+h, x:x+w]

    # 2) Crop top/bottom
    top = int(h*TOP_CROP_FRAC)
    bot = int(h*(1 - BOT_CROP_FRAC))
    board_roi = win[top:bot, :]

    # 3) Find lines → boundaries
    xs, ys = find_cell_boundaries(board_roi)

    # Map back to full image coords
    xs = [ int(x + x) for x in xs ]
    ys = [ int(y + top + y) for y in ys ]

    # 4) Extract board patch
    bx, by = xs[0], ys[0]
    board_full = full[by:ys[-1], bx:xs[-1]]

    # 5) Classify
    # Adjust xs,ys relative to board_full origin
    xs_rel = [i - xs[0] for i in xs]
    ys_rel = [i - ys[0] for i in ys]
    matrix = classify_cells(board_full, xs_rel, ys_rel, grid_bg)

    # 6) Draw & show
    viz = full.copy()
    for x in xs: cv2.line(viz, (x,ys[0]), (x,ys[-1]), (0,255,0),1)
    for y in ys: cv2.line(viz, (xs[0],y), (xs[-1],y), (0,255,0),1)
    cell_w = xs[1]-xs[0]; cell_h = ys[1]-ys[0]
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            col=(0,0,255) if matrix[i][j] else (0,255,0)
            cv2.rectangle(viz,
                (xs[j], ys[i]),
                (xs[j]+cell_w, ys[i]+cell_h),
                col,2)
    cv2.imshow("Detected Grid", viz)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 7) Save
    with open(OUTPUT_JSON,"w") as f:
        json.dump(matrix, f, indent=2)
    print(f"✅ board_matrix.json saved")
    for row in matrix: print(row)
