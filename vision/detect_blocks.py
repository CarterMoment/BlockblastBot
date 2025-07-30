#!/usr/bin/env python3
import cv2
import numpy as np
import os

from detect_board import sample_color, find_window, find_grid_bounds

# ——— CONFIG ———
ASSETS          = "assets"
SHOT            = os.path.join(ASSETS, "latest_capture.png")
UI_BG_SAMPLE    = os.path.join(ASSETS, "inverted_background.png")
GRID_BG_SAMPLE  = os.path.join(ASSETS, "inverted_empty_tile.png")

PREVIEW_OFFSET  = -5    # px above board bottom
STRIP_TILES     = 3   # how many tile‑heights to grab
BG_TOL          = 40    # BGR‐distance tolerance to consider “background”

def main():
    full = cv2.imread(SHOT)
    if full is None:
        raise FileNotFoundError(SHOT)

    # 1) locate QuickTime window + 8×8 grid
    ui_bg    = sample_color(UI_BG_SAMPLE)
    ui_x,ui_y,ui_w,ui_h = find_window(full, ui_bg)
    win      = full[ui_y:ui_y+ui_h, ui_x:ui_x+ui_w]
    grid_bg  = sample_color(GRID_BG_SAMPLE)
    gx,gy,gs,_ = find_grid_bounds(win, grid_bg)

    # 2) crop JUST enough strip under the board
    tile_h    = gs / 8.0
    top       = int(ui_y + gy + gs + PREVIEW_OFFSET)
    strip_h   = int(tile_h * STRIP_TILES)
    left      = ui_x
    right     = ui_x + ui_w
    preview   = full[top : top+strip_h, left : right]

    # 3) mask out the background color in BGR‐space
    bg_color  = ui_bg  # BGR triplet
    diff = np.linalg.norm(preview.astype(np.int32) - bg_color[None,None,:], axis=2)
    mask = (diff > BG_TOL).astype(np.uint8) * 255
    # clean it a bit
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 4) find contours → these are your pieces
    cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 5) draw outlines back on the full image
    out = full.copy()
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        cv2.rectangle(out,
            (left + x, top + y),
            (left + x + w, top + y + h),
            (0,0,255), 2)

    # 6) show everything
    cv2.imshow("Preview", preview)
    cv2.imshow("Mask",    mask)
    cv2.imshow("Outlines",out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
