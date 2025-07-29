#!/usr/bin/env python3
import cv2, numpy as np, json, os

# ——— CONFIGURATION ———
ASSETS_DIR          = "assets"
SCREENSHOT_PATH     = os.path.join(ASSETS_DIR, "latest_capture.png")
UI_BG_SAMPLE_PATH   = os.path.join(ASSETS_DIR, "inverted_background.png")
GRID_BG_SAMPLE_PATH = os.path.join(ASSETS_DIR, "inverted_empty_tile.png")
OUTPUT_JSON         = "board_matrix.json"

# Tolerances & scales
UI_BG_TOL        = 30    # BGR‑distance to detect UI background
GRID_BG_TOL      = 20    # BGR‑distance to detect grid background
MORPH_UI         = 25    # Closing kernel for window mask
MORPH_GRID       = 7     # Closing kernel for grid mask
PATCH_SCALE      = 0.6   # sample central 60% of each cell
OCCUPANCY_THRESH = 0.10  # if >10% patch pixels ≠ grid‑bg → filled

GRID_SIZE        = 8

# ——— HELPERS ———
def sample_color_bgr(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot load sample: {path}")
    h, w = img.shape[:2]
    return img[h//2, w//2].astype(np.int32)

def find_window_roi(img, ui_bg_color):
    """Mask UI background, invert, close, largest contour → window ROI."""
    diff = np.linalg.norm(img.astype(np.int32) - ui_bg_color[None,None,:], axis=2)
    mask_bg = (diff < UI_BG_TOL).astype(np.uint8) * 255
    mask_win = cv2.bitwise_not(mask_bg)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (MORPH_UI, MORPH_UI))
    mask_win = cv2.morphologyEx(mask_win, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask_win, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("Game window not detected")
    x,y,w,h = cv2.boundingRect(max(contours, key=lambda c: cv2.contourArea(c)))
    return x, y, w, h

def find_board_roi(win_img, grid_bg_color):
    """Within window, mask grid background & invert → board ROI."""
    diff = np.linalg.norm(win_img.astype(np.int32) - grid_bg_color[None,None,:], axis=2)
    mask_bg = (diff < GRID_BG_TOL).astype(np.uint8) * 255
    mask_board = cv2.bitwise_not(mask_bg)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (MORPH_GRID, MORPH_GRID))
    mask_board = cv2.morphologyEx(mask_board, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask_board, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("Board region not detected")
    x,y,w,h = cv2.boundingRect(max(contours, key=lambda c: cv2.contourArea(c)))
    side = min(w,h)
    # center square in the detected bbox
    return x + (w-side)//2, y + (h-side)//2, side, side

def extract_matrix_and_viz(full_img, ui_roi, board_roi, grid_bg_color):
    x0,y0,w0,h0 = ui_roi
    bx,by,bs,_ = board_roi

    # Prepare visualization
    viz = full_img.copy()
    cv2.rectangle(viz, (x0,y0), (x0+w0,y0+h0), (255,0,0), 2)  # blue = window
    cv2.rectangle(viz, (x0+bx,y0+by), (x0+bx+bs,y0+by+bs), (0,255,255), 2)  # yellow = board

    board = full_img[y0+by:y0+by+bs, x0+bx:x0+bx+bs]
    cell = bs // GRID_SIZE
    patch = int(cell * PATCH_SCALE)
    offset = (cell - patch)//2

    matrix = []
    for i in range(GRID_SIZE):
        row=[]
        for j in range(GRID_SIZE):
            # Extract central patch
            px = x0+bx + j*cell + offset
            py = y0+by + i*cell + offset
            patch_img = full_img[py:py+patch, px:px+patch].astype(np.int32)

            # Fraction of pixels not matching grid background
            dist = np.linalg.norm(patch_img - grid_bg_color[None,None,:], axis=2)
            non_bg_frac = np.mean(dist > GRID_BG_TOL)

            filled = 1 if non_bg_frac > OCCUPANCY_THRESH else 0
            row.append(filled)

            # Overlay red/green
            color = (0,0,255) if filled else (0,255,0)
            cv2.rectangle(viz,
                (x0+bx + j*cell,    y0+by + i*cell),
                (x0+bx + (j+1)*cell, y0+by + (i+1)*cell),
                color, 2)
        matrix.append(row)

    return matrix, viz

# ——— MAIN ———
if __name__=="__main__":
    img = cv2.imread(SCREENSHOT_PATH)
    if img is None:
        raise FileNotFoundError(f"Cannot load screenshot: {SCREENSHOT_PATH}")

    ui_bg_color   = sample_color_bgr(UI_BG_SAMPLE_PATH)
    grid_bg_color = sample_color_bgr(GRID_BG_SAMPLE_PATH)

    # Step 1: window ROI
    win_roi = find_window_roi(img, ui_bg_color)

    # Step 2: board ROI inside window
    # Extract window image region:
    x0,y0,w0,h0 = win_roi
    win_img = img[y0:y0+h0, x0:x0+w0]
    board_roi = find_board_roi(win_img, grid_bg_color)

    # Step 3: matrix + visualization
    matrix, viz = extract_matrix_and_viz(img, win_roi, board_roi, grid_bg_color)

    # Save JSON
    with open(OUTPUT_JSON, "w") as f:
        json.dump(matrix, f, indent=2)
    print(f"✅ Saved board matrix to {OUTPUT_JSON}")
    for row in matrix:
        print(row)

    # Show overlay
    cv2.imshow("Board Detection", viz)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
