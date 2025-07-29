#!/usr/bin/env python3
import cv2, numpy as np, json, os

# ——— CONFIGURATION ———
ASSETS      = "assets"
CAP_IMG     = os.path.join(ASSETS, "latest_capture.png")
OUTPUT_JSON = "board_matrix.json"

GRID        = 8
WARP_SIZE   = 800        # size to warp board to (800×800)
EDGE_THRESH = 0.02       # fraction of cell pixels with edges → marked filled

# ——— HELPERS ———

def find_board_region(img):
    """Roughly locate the board by masking dark grid background."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, m = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (15,15))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x,y,w,h = cv2.boundingRect(max(cnts, key=lambda c: cv2.contourArea(c)))
    side = min(w,h)
    # Center the square if needed
    return x, y + (h-side)//2, side, side

def detect_border_lines(board):
    """Detect horizontal/vertical border lines in the cropped board image."""
    gray = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100,
                            minLineLength=board.shape[1]*0.5,
                            maxLineGap=20)
    verts, hors = [], []
    if lines is None:
        return verts, hors
    for [[x1,y1,x2,y2]] in lines:
        if abs(x1-x2) < abs(y1-y2):  # vertical
            verts.append((x1,y1,x2,y2))
        else:
            hors.append((x1,y1,x2,y2))
    return verts, hors

def pick_border(lines, key_idx, extreme='min'):
    """
    From list of lines, pick the one with extreme coordinate:
    key_idx=0 for x1 (vertical), 1 for y1 (horizontal).
    extreme='min' or 'max'.
    """
    if not lines:
        raise RuntimeError("No border lines found")
    idx = 0 if extreme=='min' else -1
    # sort by average of the key coordinate (x or y)
    lines_sorted = sorted(lines, key=lambda l: (l[key_idx] + l[(key_idx+2)%4])//2)
    return lines_sorted[0] if extreme=='min' else lines_sorted[-1]

def line_intersection(l1, l2):
    """Intersect two infinite lines given by (x1,y1,x2,y2)."""
    x1,y1,x2,y2 = l1
    x3,y3,x4,y4 = l2
    denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
    if denom == 0:
        return None
    px = ((x1*y2-y1*x2)*(x3-x4) - (x1-x2)*(x3*y4-y3*x4)) / denom
    py = ((x1*y2-y1*x2)*(y3-y4) - (y1-y2)*(x3*y4-y3*x4)) / denom
    return [int(px), int(py)]

def warp_board(img, region):
    """Crop approximate region, detect borders, and warp to WARP_SIZE×WARP_SIZE."""
    x,y,side,_ = region
    board = img[y:y+side, x:x+side]
    verts, hors = detect_border_lines(board)

    # Pick left/right and top/bottom borders
    left  = pick_border(verts, 0, 'min')
    right = pick_border(verts, 0, 'max')
    top   = pick_border(hors, 1, 'min')
    bottom= pick_border(hors, 1, 'max')

    # Compute corner intersections
    tl = line_intersection(left,  top)
    tr = line_intersection(right, top)
    br = line_intersection(right, bottom)
    bl = line_intersection(left, bottom)
    src = np.float32([tl, tr, br, bl])
    dst = np.float32([[0,0], [WARP_SIZE,0], [WARP_SIZE,WARP_SIZE], [0,WARP_SIZE]])

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(board, M, (WARP_SIZE, WARP_SIZE))
    return warped

def classify_cells(warped):
    """Split warped board into GRID×GRID and use edge density to classify."""
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray,(3,3),0), 50,150)
    cell = WARP_SIZE // GRID
    matrix = []
    vis = warped.copy()
    for i in range(GRID):
        row=[]
        for j in range(GRID):
            patch = edges[i*cell:(i+1)*cell, j*cell:(j+1)*cell]
            frac = np.mean(patch>0)
            filled = 1 if frac > EDGE_THRESH else 0
            row.append(filled)
            color = (0,0,255) if filled else (0,255,0)
            cv2.rectangle(vis,
                          (j*cell, i*cell),
                          ((j+1)*cell, (i+1)*cell),
                          color, 2)
        matrix.append(row)
    cv2.imshow("Warped & Classified", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return matrix

def main():
    img = cv2.imread(CAP_IMG)
    if img is None:
        raise FileNotFoundError(f"Cannot load {CAP_IMG}")

    region = find_board_region(img)
    warped = warp_board(img, region)
    matrix = classify_cells(warped)

    with open(OUTPUT_JSON,"w") as f:
        json.dump(matrix,f,indent=2)
    print(f"✅ Saved matrix to {OUTPUT_JSON}")
    for row in matrix:
        print(row)

if __name__=="__main__":
    main()
