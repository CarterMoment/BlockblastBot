#!/usr/bin/env python3
import cv2, numpy as np, json, os

# ——— CONFIG ———
ASSETS_DIR    = "assets"
CAPTURE_IMG   = os.path.join(ASSETS_DIR, "latest_capture.png")
BG_SAMPLE_IMG = os.path.join(ASSETS_DIR, "inverted_background.png")
OUTPUT_JSON   = "board_matrix.json"

GRID_SIZE     = 8
BG_THRESH     = 60      # gray threshold to isolate board region
MORPH_SIZE    = 15
PATCH_SCALE   = 0.5     # sample inner 50% of each cell
SAT_THRESHOLD = 60      # if mean S > this, mark as filled

# ——— HELPERS ———
def find_board_region(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, m = cv2.threshold(gray, BG_THRESH, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (MORPH_SIZE, MORPH_SIZE))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        raise RuntimeError("Board region not found.")
    x, y, w, h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
    side = min(w, h)
    return x, y, side, side

def extract_by_saturation(img, region):
    x, y, side, _ = region
    board = img[y:y+side, x:x+side]
    hsv = cv2.cvtColor(board, cv2.COLOR_BGR2HSV)
    cell_w = cell_h = side // GRID_SIZE

    matrix = []
    viz = board.copy()
    for i in range(GRID_SIZE):
        row = []
        for j in range(GRID_SIZE):
            # inner patch
            x1 = int(j*cell_w + cell_w*(1-PATCH_SCALE)/2)
            y1 = int(i*cell_h + cell_h*(1-PATCH_SCALE)/2)
            x2 = x1 + int(cell_w*PATCH_SCALE)
            y2 = y1 + int(cell_h*PATCH_SCALE)
            patch = hsv[y1:y2, x1:x2]
            mean_s = patch[:,:,1].mean()

            filled = 1 if mean_s > SAT_THRESHOLD else 0
            row.append(filled)

            # draw viz: red for filled, green for empty
            color = (0,0,255) if filled else (0,255,0)
            cv2.rectangle(viz,
                          (j*cell_w, i*cell_h),
                          ((j+1)*cell_w, (i+1)*cell_h),
                          color, 2)

        matrix.append(row)
    return matrix, viz

# ——— MAIN ———
if __name__ == "__main__":
    img = cv2.imread(CAPTURE_IMG)
    if img is None:
        raise FileNotFoundError(f"Cannot load capture: {CAPTURE_IMG}")

    region = find_board_region(img)
    matrix, viz = extract_by_saturation(img, region)

    # Save JSON
    with open(OUTPUT_JSON, "w") as f:
        json.dump(matrix, f, indent=2)
    print(f"✅ Saved board matrix to {OUTPUT_JSON}")
    for row in matrix: print(row)

    # Show visualization
    cv2.imshow("Board Detection (red=filled, green=empty)", viz)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
