#!/usr/bin/env python3
import cv2, numpy as np, json, os

# ——— CONFIG ———
ASSETS_DIR     = "assets"
CAPTURE_IMG    = os.path.join(ASSETS_DIR, "latest_capture.png")
BG_SAMPLE_IMG  = os.path.join(ASSETS_DIR, "inverted_background.png")
EMPTY_SAMPLE   = os.path.join(ASSETS_DIR, "inverted_empty_tile.png")
OUTPUT_JSON    = "board_matrix.json"

GRID_SIZE      = 8
BG_THRESH      = 60       # grey threshold for "board area" mask
MORPH_SIZE     = 15
PATCH_SCALE    = 0.5      # sample inner patch (50%)
SAT_THRESH     = 60       # saturation threshold to consider a pixel "filled"
PIXEL_FRAC     = 0.5      # fraction of pixels above SAT_THRESH to call "filled"

# ——— HELPERS ———
def load_bg_color(path):
    """Load center pixel BGR of background sample (unused here, but kept for fallback)"""
    img = cv2.imread(path)
    h,w = img.shape[:2]
    return img[h//2, w//2]

def find_board_region(img):
    H,W = img.shape[:2]
    # 1) Restrict to middle half height
    y0, y1 = H//4, 3*H//4
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, BG_THRESH, 255, cv2.THRESH_BINARY_INV)
    roi = np.zeros_like(mask)
    roi[y0:y1, :] = mask[y0:y1, :]

    # 2) Morph close to fill small gaps
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (MORPH_SIZE, MORPH_SIZE))
    roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, k)

    # 3) Largest contour
    cnts, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        raise RuntimeError("Board region not found!")
    x,y,w,h = cv2.boundingRect(max(cnts, key=lambda c: cv2.contourArea(c)))

    # 4) Square-ize
    side = min(w,h)
    return x, y + (h-side)//2, side, side  # center vertically in that bbox

def build_matrix(img, region):
    x,y,side,_ = region
    board = img[y:y+side, x:x+side]
    hsv = cv2.cvtColor(board, cv2.COLOR_BGR2HSV)
    cell_w = cell_h = side // GRID_SIZE

    mat = []
    for i in range(GRID_SIZE):
        row=[]
        for j in range(GRID_SIZE):
            # inner patch
            xx = int(j*cell_w + (cell_w*(1-PATCH_SCALE)/2))
            yy = int(i*cell_h + (cell_h*(1-PATCH_SCALE)/2))
            pw = int(cell_w*PATCH_SCALE); ph = int(cell_h*PATCH_SCALE)
            patch = hsv[yy:yy+ph, xx:xx+pw]

            # compute fraction of pixels with S > SAT_THRESH
            sat = patch[:,:,1]
            filled_frac = np.mean(sat > SAT_THRESH)
            row.append(1 if filled_frac > PIXEL_FRAC else 0)
        mat.append(row)
    return mat

def main():
    img = cv2.imread(CAPTURE_IMG)
    if img is None:
        raise FileNotFoundError(f"Cannot load {CAPTURE_IMG}")

    region = find_board_region(img)
    matrix = build_matrix(img, region)

    with open(OUTPUT_JSON, "w") as f:
        json.dump(matrix, f, indent=2)
    print(f"✅ Saved board matrix to {OUTPUT_JSON}")
    for r in matrix:
        print(r)

    # visualize
    vis = img.copy()
    x,y,side,_ = region
    cv2.rectangle(vis, (x,y), (x+side,y+side), (255,0,0), 2)
    cw = ch = side//GRID_SIZE
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            color = (0,0,255) if matrix[i][j] else (0,255,0)
            cv2.rectangle(vis,
                          (x+j*cw,   y+i*ch),
                          (x+(j+1)*cw, y+(i+1)*ch),
                          color, 1)
    cv2.imshow("Detect Board", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()
