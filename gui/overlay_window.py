
#!/usr/bin/env python3
import tkinter as tk
import json
import os
import threading
import time

RECOMMENDED = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "recommended_move.json")
)
POLL_INTERVAL_MS = 1000  # reload every second

# color mapping: 0=empty, 1=existing, 2=recommended
COLORS = {
    0: "white",
    1: "lightgray",
    2: "green"
}

class RecommendationViewer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("BlockBlast – Next Move")
        self.cell_size = 50

        # load once to get size
        mat = self._load_matrix()
        self.N = len(mat)
        canvas_sz = self.N * self.cell_size
        self.canvas = tk.Canvas(self, width=canvas_sz, height=canvas_sz)
        self.canvas.pack()
        self._current = None

        # start polling in the Tk event loop
        self.after(0, self._poll)

    def _load_matrix(self):
        with open(RECOMMENDED) as f:
            data = json.load(f)
        return data.get("initial_with_move", [])

    def _draw(self, matrix):
        self.canvas.delete("all")
        for i, row in enumerate(matrix):
            for j, val in enumerate(row):
                x0 = j * self.cell_size
                y0 = i * self.cell_size
                x1 = x0 + self.cell_size
                y1 = y0 + self.cell_size
                self.canvas.create_rectangle(
                    x0, y0, x1, y1,
                    fill=COLORS.get(val, "white"),
                    outline="black"
                )

    def _poll(self):
        try:
            mat = self._load_matrix()
            if mat != self._current:
                self._current = mat
                self._draw(mat)
        except FileNotFoundError:
            pass
        finally:
            # schedule next poll
            self.after(POLL_INTERVAL_MS, self._poll)

def main():
    app = RecommendationViewer()
    app.mainloop()

if __name__ == "__main__":
    main()
