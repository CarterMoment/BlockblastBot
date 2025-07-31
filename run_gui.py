#!/usr/bin/env python3
import os
import sys
import time
import subprocess

# ─── CONFIGURATION ─────────────────────────────────────────────────────
# How many seconds between re-runs of detect→simulate
REFRESH_INTERVAL = 1.0

# Relative paths (from this file) to your scripts
VISION_DIR        = os.path.join(os.path.dirname(__file__), "vision")
LOGIC_DIR        = os.path.join(os.path.dirname(__file__), "logic")
DETECT_BOARD_PY   = os.path.join(VISION_DIR, "detect_board.py")
DETECT_BLOCKS_PY  = os.path.join(VISION_DIR, "detect_blocks.py")
SIMULATOR_PY      = os.path.join(LOGIC_DIR, "move_simulator.py")
GUI_SCRIPT        = os.path.join(os.path.dirname(__file__), "gui", "overlay_window.py")
# ─────────────────────────────────────────────────────────────────────────

def main():
    # 1) launch GUI (it polls recommended_move.json)
    gui_proc = subprocess.Popen([sys.executable, GUI_SCRIPT])
    print(f"[pipeline] GUI started (pid={gui_proc.pid})")

    try:
        while True:
            # 2) detect the board
            print("[pipeline] Running detect_board…")
            subprocess.run([sys.executable, DETECT_BOARD_PY], check=True)

            # 3) detect the next blocks
            print("[pipeline] Running detect_blocks…")
            subprocess.run([sys.executable, DETECT_BLOCKS_PY], check=True)

            # 4) simulate moves
            print("[pipeline] Running move_simulator…")
            subprocess.run([sys.executable, SIMULATOR_PY], check=True)

            # 5) wait before next cycle
            time.sleep(REFRESH_INTERVAL)

    except KeyboardInterrupt:
        print("\n[pipeline] Interrupted by user, shutting down…")

    finally:
        # terminate GUI
        gui_proc.terminate()
        gui_proc.wait()
        print("[pipeline] GUI terminated. Goodbye.")

if __name__ == "__main__":
    main()