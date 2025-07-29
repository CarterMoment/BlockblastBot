#!/usr/bin/env python3
import cv2, numpy as np, json, os

# ——— CONFIGURATION ———
ASSETS_DIR      = "assets"
CAPTURE_IMG     = os.path.join(ASSETS_DIR, "latest_capture.png")
BG_SAMPLE_IMG   = os.path.join(ASSETS_DIR, "inverted_background.png")
OUTPUT_JSON     = "board_matrix.json"

GRID_SIZE       = 8
BG_TOLERANCE    = 20     # Max BGR‐distance to consider “background” (brown) 
MORPH_SIZE      = 15     # Morph kernel to close holes in the grid mask
PATCH_SCALE     = 0.5    # Sample central 50%×50% patch of each cell
OCCUPANCY_THRESH= 0.20   # If >20% of patch pixels are non‐background → filled

# ——— HELPERS ———
def sample_bg_color(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    h, w = img.shape[:2]
    return img[h//2, w//2].astype(np.int32)

def find_board_region(img, bg_color):
    """Mask background, invert, close holes, find largest contour."""
    diff = np.linalg.norm(img.astype(np.int32) - bg_color[None,None,:], axis=2)
    bg_mask = (diff < BG_TOLERANCE).astype(np.uint8) * 255
    board_mask = cv2.bitwise_not(bg_mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (MORPH_SIZE, MORPH_SIZE))
    board_mask = cv2.morphologyEx(board_mask, cv2.MORPH_CLOSE, kernel)
    cnts, _ = cv2.findContours(board_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        raise RuntimeError("Could not find board!")
    x,y,w,h = cv2.boundingRect(max(cnts, key=lambda c: cv2.contourArea(c)))
    side = min(w,h)
    return x, y, side, side

def extract_board_matrix(img, bg_color):
    x,y,side,_ = find_board_region(img, bg_color)
    board = img[y:y+side, x:x+side]
    cell = side // GRID_SIZE
    patch = int(cell * PATCH_SCALE)
    offset = (cell - patch) // 2

    matrix = []
    vis = board.copy()
    for i in range(GRID_SIZE):
        row=[]
        for j in range(GRID_SIZE):
            x1 = j*cell + offset
            y1 = i*cell + offset
            x2, y2 = x1 + patch, y1 + patch
            tile = board[y1:y2, x1:x2].astype(np.int32)

            # count non‐background pixels
            dist = np.linalg.norm(tile - bg_color[None,None,:], axis=2)
            non_bg = np.count_nonzero(dist > BG_TOLERANCE)
            frac = non_bg / (patch*patch)
            filled = 1 if frac > OCCUPANCY_THRESH else 0
            row.append(filled)

            # draw viz
            color = (0,0,255) if filled else (0,255,0)
            cv2.rectangle(vis, (j*cell, i*cell), ((j+1)*cell, (i+1)*cell), color, 2)
        matrix.append(row)

    # overlay board crop on full image for reference
    full_vis = img.copy()
    cv2.rectangle(full_vis, (x,y), (x+side,y+side), (255,0,0), 2)
    full_vis[y:y+side, x:x+side] = vis

    cv2.imshow("Board Detection (red=filled/green=empty)", full_vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return matrix

# ——— MAIN ———
if __name__=="__main__":
    cap = cv2.imread(CAPTURE_IMG)
    if cap is None:
        raise FileNotFoundError(f"Cannot load {CAPTURE_IMG}")
    bg_col = sample_bg_color(BG_SAMPLE_IMG)

    mat = extract_board_matrix(cap, bg_col)

    with open(OUTPUT_JSON, "w") as f:
        json.dump(mat, f, indent=2)
    print(f"✅ Saved matrix to {OUTPUT_JSON}")
    for row in mat:
        print(row)
