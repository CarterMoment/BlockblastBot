# Block Blast Bot

A computer vision–powered bot for the mobile puzzle game **Block Blast**.  
This project detects the game board and block pieces in real time, simulates possible moves, and outputs recommended placements to maximize score. It combines **OpenCV**, **Python**, and custom game logic to create an automated assistant for gameplay.

---

## Features

- 🎮 **Board Detection**  
  Captures an 8x8 Block Blast board and converts it into a matrix (`0` = empty, `1` = filled).  

- 🧩 **Block Recognition**  
  Detects the three available blocks at the bottom of the screen.  

- 🤖 **Move Simulation**  
  Runs algorithms to calculate the best move sequence.  

- ⏱ **Pipeline Reliability**  
  Automatically retries detection if the pipeline hangs for more than 10 seconds (e.g., during animations).  

- 📂 **JSON Outputs**  
  - `board_matrix.json` → Current board state  
  - `next_blocks.json` → Next available blocks  
  - `recommended_move.json` → Bot’s chosen move  

---

## File Structure

project-root/
│
├── vision/ # Computer vision scripts
│ ├── detect_board.py # Extracts 8x8 board matrix
│ ├── detect_blocks.py # Recognizes available blocks
│ ├── capture.py # Capture Block Blast window
│
├── logic/ # Move evaluation and simulation
│
├── gui/ # Optional visualization / user interface
│
├── assets/ # Reference images (e.g., board anchor, backgrounds)
│
├── latest_capture.png # Last captured screenshot
├── board_matrix.json # Saved board matrix
├── next_blocks.json # Saved block set
├── recommended_move.json # Bot’s recommended move
└── requirements.txt # Python dependencies

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

2. Capture the game window

The bot uses QuickTime screen sharing to mirror the device onto your computer.

Open QuickTime and start a new recording of your device screen.

Ensure the Block Blast window is visible.

The bot will use board_anchor.png to locate the game region.

3. Run the bot

Example:

python vision/detect_board.py
python vision/detect_blocks.py
python logic/run_pipeline.py


This will generate JSON files with the current board state, blocks, and recommended move.

Usage

Update the board capture:

python vision/capture.py


Detect the board matrix:

python vision/detect_board.py


Detect the available blocks:

python vision/detect_blocks.py


Get a recommended move:
python logic/run_pipeline.py
