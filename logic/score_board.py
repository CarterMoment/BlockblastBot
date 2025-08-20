# score_board.py

import json
from collections import deque

# ─── SCORING WEIGHTS ─────────────────────────────────────────────────
FULL_LINE_REWARD = 10   # reward per full row or column
EDGE_REWARD      = 1    # reward per block on an edge (non-corner)
CORNER_REWARD    = 2    # reward per block in a corner
HOLE_PENALTY     = 5    # penalty per enclosed hole cell
# ─────────────────────────────────────────────────────────────────────

def clear_full_lines(board):
    """Return a copy of board with any full row/column of 1s set to 0."""
    h, w = len(board), len(board[0])
    b = [row[:] for row in board]
    while True:
        full_rows = [i for i,row in enumerate(b) if all(cell == 1 for cell in row)]
        full_cols = [j for j in range(w)    if all(b[i][j] == 1 for i in range(h))]
        if not full_rows and not full_cols:
            break
        for r in full_rows:
            for j in range(w):
                b[r][j] = 0
        for c in full_cols:
            for i in range(h):
                b[i][c] = 0
    return b

def count_holes(board):
    """
    Count zero‐cells that form regions not touching the border.
    """
    h, w = len(board), len(board[0])
    seen = [[False]*w for _ in range(h)]
    holes = 0
    for i in range(h):
        for j in range(w):
            if board[i][j] == 0 and not seen[i][j]:
                q = deque([(i,j)])
                seen[i][j] = True
                region = [(i,j)]
                touches_border = (i in (0,h-1) or j in (0,w-1))
                while q:
                    x,y = q.popleft()
                    for dx,dy in ((1,0),(-1,0),(0,1),(0,-1)):
                        nx, ny = x+dx, y+dy
                        if 0 <= nx < h and 0 <= ny < w and not seen[nx][ny] and board[nx][ny] == 0:
                            seen[nx][ny] = True
                            q.append((nx,ny))
                            region.append((nx,ny))
                            if nx in (0,h-1) or ny in (0,w-1):
                                touches_border = True
                if not touches_border:
                    holes += len(region)
    return holes

def get_score(board):
    """
    Score an 8×8 board (list of lists of 0/1):
      1) Clear full rows/columns → reward FULL_LINE_REWARD each.
      2) On the cleared board, for each 1:
         - if in a corner: +CORNER_REWARD
         - elif on an edge: +EDGE_REWARD
      3) Count enclosed hole‐cells and subtract HOLE_PENALTY each.
    Returns a single integer score (higher is better).
    """
    # 1) reward full lines
    h, w = len(board), len(board[0])
    full_rows = [i for i,row in enumerate(board) if all(cell==1 for cell in row)]
    full_cols = [j for j in range(w)    if all(board[i][j]==1 for i in range(h))]
    score = FULL_LINE_REWARD * (len(full_rows) + len(full_cols))

    # 2) apply clears, then reward edge/corner blocks
    cleared = clear_full_lines(board)
    for i in range(h):
        for j in range(w):
            if cleared[i][j] == 1:
                if (i in (0,h-1)) and (j in (0,w-1)):
                    score += CORNER_REWARD
                elif i in (0,h-1) or j in (0,w-1):
                    score += EDGE_REWARD

    # 3) subtract hole penalties
    hole_cells = count_holes(cleared)
    score -= HOLE_PENALTY * hole_cells

    return score

# Optional CLI for quick testing
if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv)>1 else "board_matrix.json"
    board = json.load(open(path))
    print(get_score(board))