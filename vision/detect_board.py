#!/usr/bin/env python3
import cv2, numpy as np, json, os

# ——— CONFIGURATION ———
ASSETS_DIR    = "assets"
CAPTURE_IMG   = os.path.join(ASSETS_DIR, "latest_capture.png")
BG_SAMPLE_IMG = os.path.join(ASSETS_DIR, "inverted_background.png")
OUTPUT_JSON   = "board_matrix.json"

GRID_SIZE     = 8
BG_TOLERANCE  = 30    # how different a cell must be from background to count as "filled"
MORPH_SIZE    = 15    # morphological kernel size for closing the board mask
PATCH_SCALE   = 0.5   # fraction of cell width/height to sample (inner patch)

# ——— UTILITIES ———
def load_background_color(path):
    """Load the sample background image and return its center BGR color."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot load background sample: {path}")
    h, w = img.shape[:2]
    return img[h//2, w//2].astype(np.float32)

def find_board_region(img, bg_color):
    """
    Create a mask of pixels *close* to bg_color, invert it, close holes,
    and return the bbox of the largest contour as (x,y,w,h) square.
    """
    # Compute per‑pixel distance from bg_color
    diff = np.linalg.norm(img.astype(np.float32) - bg_color, axis=2)
    mask_bg = (diff < BG_TOLERANCE).astype(np.uint8) * 255    # 255 = background
    mask = cv2.bitwise_not(mask_bg)                          # invert → board+blocks

    # Close small holes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (MORPH_SIZE, MORPH_SIZE))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Find the largest contour
    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        raise RuntimeError("Could not find board region!")
    x, y, w, h = cv2.boundingRect(max(cnts, key=cv2.contourArea))

    # Make square by cropping to min(w,h)
    side = min(w, h)
    return x, y, side, side

def extract_board_matrix(img, bg_color):
    """Given the full RGB image and bg_color, return 8×8 matrix of 0/1."""
    x, y, side, _ = find_board_region(img, bg_color)
    board = img[y:y+side, x:x+side]
    cell_w = cell_h = side // GRID_SIZE

    matrix = []
    for i in range(GRID_SIZE):
        row = []
        for j in range(GRID_SIZE):
            # inner patch (to avoid border/shadows)
            px = int(j*cell_w + (cell_w*(1-PATCH_SCALE)/2))
            py = int(i*cell_h + (cell_h*(1-PATCH_SCALE)/2))
            pw = int(cell_w * PATCH_SCALE)
            ph = int(cell_h * PATCH_SCALE)

            patch = board[py:py+ph, px:px+pw].astype(np.float32)
            avg_color = patch.reshape(-1,3).mean(axis=0)

            dist = np.linalg.norm(avg_color - bg_color)
            row.append(1 if dist > BG_TOLERANCE else 0)
        matrix.append(row)
    return matrix

# ——— MAIN ———
if __name__ == "__main__":
    # Load images
    cap = cv2.imread(CAPTURE_IMG)
    if cap is None:
        raise FileNotFoundError(f"Cannot load capture image: {CAPTURE_IMG}")

    bg_col = load_background_color(BG_SAMPLE_IMG)

    # Extract & save
    mat = extract_board_matrix(cap, bg_col)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(mat, f, indent=2)

    print(f"✅ Saved board matrix to {OUTPUT_JSON}")
    for row in mat:
        print(row)
