#!/usr/bin/env python3
import json
import math
from score_board import clear_full_lines, get_score

BOARD_JSON   = "board_matrix.json"
BLOCKS_JSON  = "next_blocks.json"
OUTPUT_JSON  = "recommended_move.json"

def load_board(path=BOARD_JSON):
    with open(path) as f:
        return json.load(f)

def load_blocks(path=BLOCKS_JSON):
    with open(path) as f:
        data = json.load(f)
    return [blk["matrix"] for blk in data]

def get_valid_positions(board, shape):
    H, W = 8, 8
    h, w = len(shape), len(shape[0])
    pos = []
    for r in range(H - h + 1):
        for c in range(W - w + 1):
            ok = True
            for dr in range(h):
                for dc in range(w):
                    if shape[dr][dc] and board[r+dr][c+dc]:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                pos.append((r, c))
    return pos

def place_and_clear(board, shape, position):
    r0, c0 = position
    newb = [row[:] for row in board]
    for dr, row in enumerate(shape):
        for dc, v in enumerate(row):
            if v:
                newb[r0+dr][c0+dc] = 1
    return clear_full_lines(newb)

def find_best_sequence(board, blocks):
    best = {"score": -math.inf, "sequence": []}
    N = len(blocks)
    remaining = set(range(N))

    def dfs(cur_board, rem, seq):
        if not rem:
            sc = get_score(cur_board)
            if sc > best["score"]:
                best["score"], best["sequence"] = sc, seq.copy()
            return
        for i in list(rem):
            shape = blocks[i]
            for pos in get_valid_positions(cur_board, shape):
                nxt = place_and_clear(cur_board, shape, pos)
                seq.append((i, pos))
                dfs(nxt, rem - {i}, seq)
                seq.pop()

    dfs(board, remaining, [])
    return best

def overlay_first_move(board, shape, position):
    """
    Returns a new 8×8 matrix where:
      - existing 1s stay as 1
      - zeros stay 0
      - cells covered by shape at position become 2
    """
    overlay = [row[:] for row in board]
    r0, c0 = position
    for dr, row in enumerate(shape):
        for dc, v in enumerate(row):
            if v:
                overlay[r0+dr][c0+dc] = 2
    return overlay

def main():
    board  = load_board()
    blocks = load_blocks()
    best   = find_best_sequence(board, blocks)

    result = {
        "initial_with_move": None,
        "sequence": best["sequence"],
        "score": best["score"]
    }

    if best["sequence"]:
        first_blk, first_pos = best["sequence"][0]
        result["initial_with_move"] = overlay_first_move(
            board, blocks[first_blk], first_pos
        )
    else:
        # no valid moves
        result["initial_with_move"] = [row[:] for row in board]

    # save out for your GUI to consume
    with open(OUTPUT_JSON, "w") as f:
        json.dump(result, f, indent=2)

    # also print for debugging
    print(json.dumps(result, indent=2))

if __name__=="__main__":
    main()