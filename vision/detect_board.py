#!/usr/bin/env python3
import cv2, numpy as np, json, os

# ——— CONFIGURATION ———
ASSETS_DIR          = "assets"
SCREENSHOT_PATH     = os.path.join(ASSETS_DIR, "latest_capture.png")
UI_BG_SAMPLE_PATH   = os.path.join(ASSETS_DIR, "inverted_background.png")
GRID_BG_SAMPLE_PATH = os.path.join(ASSETS_DIR, "inverted_empty_tile.png")
OUTPUT_JSON         = "board_matrix.json"

GRID_SIZE           = 8

UI_BG_TOL            = 30    # tolerance for UI (QuickTime) background sampling
GRID_BG_TOL          = 20    # tolerance for grid‑background (empty cell) sampling

MORPH_UI             = 25    # kernel size to close holes in window mask
MORPH_GRID           = 7     # kernel size to close holes in grid mask

PATCH_SCALE          = 0.6   # sample central 60% patch of each cell
OCCUPANCY_THRESH     = 0.10  # if >10% of patch pixels ≠ grid‑bg → filled

# ** Adjust these insets to exclude UI chrome that’s cutting off rows/columns **
INSET_TOP_FRAC       = 0.10  # crop away top 10% of window (hides number overlay)
INSET_BOTTOM_FRAC    = 0.00  # crop away bottom 0%   (keeps last rows)
INSET_LEFT_FRAC      = 0.08  # crop away left 8%
INSET_RIGHT_FRAC     = 0.08  # crop away right 8%


def sample_color_bgr(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    h, w = img.shape[:2]
    return img[h//2, w//2].astype(np.int32)


def find_window_roi(img, ui_bg_color):
    diff   = np.linalg.norm(img.astype(np.int32) - ui_bg_color[None,None,:], axis=2)
    mask_bg= (diff < UI_BG_TOL).astype(np.uint8)*255
    mask_w = cv2.bitwise_not(mask_bg)
    kern   = cv2.getStructuringElement(cv2.MORPH_RECT, (MORPH_UI, MORPH_UI))
    mask_w = cv2.morphologyEx(mask_w, cv2.MORPH_CLOSE, kern)

    cnts, _ = cv2.findContours(mask_w, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        raise RuntimeError("Game window not detected")
    x,y,w,h = cv2.boundingRect(max(cnts, key=lambda c: cv2.contourArea(c)))
    return x, y, w, h


def find_board_roi(win_img, grid_bg_color):
    H, W = win_img.shape[:2]

    # inset margins to ignore UI chrome
    x1 = int(W * INSET_LEFT_FRAC)
    x2 = int(W * (1 - INSET_RIGHT_FRAC))
    y1 = int(H * INSET_TOP_FRAC)
    y2 = int(H * (1 - INSET_BOTTOM_FRAC))
    crop = win_img[y1:y2, x1:x2]

    diff   = np.linalg.norm(crop.astype(np.int32) - grid_bg_color[None,None,:], axis=2)
    mask_bg= (diff < GRID_BG_TOL).astype(np.uint8)*255
    mask_b = cv2.bitwise_not(mask_bg)
    kern   = cv2.getStructuringElement(cv2.MORPH_RECT, (MORPH_GRID, MORPH_GRID))
    mask_b = cv2.morphologyEx(mask_b, cv2.MORPH_CLOSE, kern)

    cnts, _ = cv2.findContours(mask_b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        raise RuntimeError("Board region not found after insets")

    x, y, w, h = cv2.boundingRect(max(cnts, key=lambda c: cv2.contourArea(c)))
    side = min(w, h)

    # map back into window coordinates
    return x + x1, y + y1, side, side


def extract_matrix_and_viz(full_img, ui_roi, board_roi, grid_bg_color):
    x0, y0, w0, h0 = ui_roi
    bx, by, bs, _  = board_roi

    viz = full_img.copy()
    # blue = window, yellow = board
    cv2.rectangle(viz, (x0, y0), (x0 + w0, y0 + h0), (255, 0, 0), 2)
    cv2.rectangle(viz,
                  (x0 + bx, y0 + by),
                  (x0 + bx + bs, y0 + by + bs),
                  (0, 255, 255), 2)

    cell_size = bs // GRID_SIZE
    patch     = int(cell_size * PATCH_SCALE)
    offset    = (cell_size - patch) // 2

    matrix = []
    for i in range(GRID_SIZE):
        row = []
        for j in range(GRID_SIZE):
            px = x0 + bx + j*cell_size + offset
            py = y0 + by + i*cell_size + offset
            patch_img = full_img[py:py+patch, px:px+patch].astype(np.int32)

            dist = np.linalg.norm(patch_img - grid_bg_color[None,None,:], axis=2)
            non_bg_frac = np.mean(dist > GRID_BG_TOL)

            filled = 1 if non_bg_frac > OCCUPANCY_THRESH else 0
            row.append(filled)

            color = (0, 0, 255) if filled else (0, 255, 0)
            cv2.rectangle(viz,
                (x0 + bx + j*cell_size,    y0 + by + i*cell_size),
                (x0 + bx + (j+1)*cell_size, y0 + by + (i+1)*cell_size),
                color, 2)
        matrix.append(row)

    return matrix, viz


if __name__ == "__main__":
    full = cv2.imread(SCREENSHOT_PATH)
    if full is None:
        raise FileNotFoundError(f"Cannot load screenshot: {SCREENSHOT_PATH}")

    ui_bg   = sample_color_bgr(UI_BG_SAMPLE_PATH)
    grid_bg = sample_color_bgr(GRID_BG_SAMPLE_PATH)

    ui_roi    = find_window_roi(full, ui_bg)
    x0, y0, w0, h0 = ui_roi
    win_img   = full[y0:y0+h0, x0:x0+w0]

    board_roi = find_board_roi(win_img, grid_bg)
    matrix, viz = extract_matrix_and_viz(full, ui_roi, board_roi, grid_bg)

    with open(OUTPUT_JSON, "w") as f:
        json.dump(matrix, f, indent=2)
    print(f"✅ Saved board matrix to {OUTPUT_JSON}")
    for row in matrix:
        print(row)

    cv2.imshow("Board Detection", viz)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
