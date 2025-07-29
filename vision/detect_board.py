#!/usr/bin/env python3
import cv2, numpy as np, json, os

# ——— CONFIGURATION ———
ASSETS_DIR       = "assets"
CAPTURE_IMG      = "latest_capture.png"
EMPTY_TILE_IMG   = os.path.join(ASSETS_DIR, "inverted_empty_tile.png")
OUTPUT_JSON      = "board_matrix.json"

GRID_SIZE        = 8
BG_THRESH        = 60     # gray threshold to find the board mask
MORPH_SIZE       = 15     # closes small gaps in grid mask
PATCH_SCALE      = 0.5    # sample central 50% patch of each cell
EMPTY_HSV_TOL    = np.array([10, 60, 60])  # tolerance in HSV for matching "empty" cell
EMPTY_FRAC_THRESH = 0.5   # >50% of pixels match empty → mark empty

# ——— HELPERS ———
def load_empty_hsv(path):
    """Return the center HSV of the provided empty-tile image."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, w = hsv.shape[:2]
    return hsv[h//2, w//2]

def find_board_region(img):
    """Find the square board region by masking dark grid background."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, BG_THRESH, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (MORPH_SIZE, MORPH_SIZE))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        raise RuntimeError("Could not locate board!")
    x,y,w,h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
    side = min(w,h)
    return x, y, side, side

def extract_matrix(img, empty_hsv):
    """Split the board into 8x8 cells and classify by matching empty_hsv."""
    x,y,side,_ = find_board_region(img)
    board = img[y:y+side, x:x+side]
    hsv_board = cv2.cvtColor(board, cv2.COLOR_BGR2HSV)
    cell = side // GRID_SIZE
    patch = int(cell * PATCH_SCALE)
    offset = (cell - patch)//2

    matrix = []
    vis = img.copy()
    # overlay board crop
    cv2.rectangle(vis, (x,y), (x+side, y+side), (255,0,0), 2)

    for i in range(GRID_SIZE):
        row = []
        for j in range(GRID_SIZE):
            x1 = x + j*cell + offset
            y1 = y + i*cell + offset
            patch_hsv = hsv_board[i*cell+offset:i*cell+offset+patch,
                                  j*cell+offset:j*cell+offset+patch]

            # compute fraction of pixels within EMPTY_HSV_TOL tolerance
            diff = np.abs(patch_hsv.astype(int) - empty_hsv[None,None,:])
            within = np.all(diff <= EMPTY_HSV_TOL, axis=2)
            frac = np.mean(within)

            empty = frac >= EMPTY_FRAC_THRESH
            row.append(0 if empty else 1)

            # draw overlay: green=empty, red=filled
            color = (0,255,0) if empty else (0,0,255)
            cv2.rectangle(vis,
                (x + j*cell, y + i*cell),
                (x + (j+1)*cell, y + (i+1)*cell),
                color, 2)
        matrix.append(row)

    # show annotation
    cv2.imshow("Board Detection", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return matrix

# ——— MAIN ———
if __name__=="__main__":
    cap = cv2.imread(CAPTURE_IMG)
    if cap is None:
        raise FileNotFoundError(f"Cannot load capture: {CAPTURE_IMG}")

    empty_hsv = load_empty_hsv(EMPTY_TILE_IMG)
    board = extract_matrix(cap, empty_hsv)

    with open(OUTPUT_JSON, "w") as f:
        json.dump(board, f, indent=2)
    print(f"✅ Saved board matrix to {OUTPUT_JSON}")
    for row in board:
        print(row)
