# 🎯 Tic-Tac-Toe AI — Minimax Algorithm

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Tkinter](https://img.shields.io/badge/GUI-Tkinter-green?style=for-the-badge)
![Algorithm](https://img.shields.io/badge/Algorithm-Minimax-orange?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-purple?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=for-the-badge)

**CodSoft Artificial Intelligence Internship — Task 2**

*An unbeatable AI opponent powered by the Minimax algorithm, wrapped in a modern dark-themed desktop GUI.*

</div>

---

## 👨‍💻 Internship Information

| Field | Details |
|:------|:--------|
| **Developer** | Sanjeevikumar D |
| **Internship** | CodSoft Artificial Intelligence Internship |
| **Task** | Task 2 – Tic-Tac-Toe AI |
| **Domain** | Artificial Intelligence |
| **Tech Stack** | Python · Tkinter · Minimax Algorithm |

---

## 📌 Project Overview

This project implements a **fully-featured Tic-Tac-Toe game** where a human player challenges an AI opponent that **never loses**. The AI uses the classic **Minimax algorithm** to exhaustively search the entire game tree and always select the mathematically optimal move — guaranteeing a win or a draw for the AI in every game.

The application runs as a **standalone desktop GUI** built with Python's built-in `tkinter` library, requiring **zero external dependencies**.

---

## ✨ Features

### 🎮 Gameplay
- **Human vs AI** — You play `X` (Green), AI plays `O` (Red)
- **Human always moves first**
- Prevents clicking on **occupied cells** or after game ends
- **900 ms AI "thinking" delay** for a realistic experience
- Full **3×3 board** — all nine cells always visible at any window size

### 🤖 AI Intelligence
- Powered by the **Minimax algorithm**
- Exhaustive **recursive game tree search** — searches all 255,168 possible game states
- **Pure stateless evaluation** — no random moves, no GUI side-effects inside the search
- The AI is **mathematically unbeatable**

### 📊 Score Tracking
- Tracks **Human Wins**, **AI Wins**, and **Draws** across the session
- Scores persist across games until **New Match** is clicked

### 🎨 Visual Design
- **Modern dark theme** — curated colour palette
- **Golden highlight** on winning cells (`#FFD54F`)
- **Hover effects** on all interactive elements
- **Welcome overlay** with game instructions on startup
- Status messages: `Your Turn`, `AI Thinking…`, `You Win!`, `AI Wins!`, `It's a Draw!`

### ▶️ Play Again Flow
After every game ends, a **green "Play Again?" button** appears instantly in the status bar, along with a keyboard shortcut.

| Action | Result |
|:-------|:-------|
| Click **▶ Play Again?** | Reset board, keep scores |
| Press **`R`** on keyboard | Reset board, keep scores |
| Click **🔄 Restart Game** | Reset board, keep scores |
| Click **🆕 New Match** | Reset board AND reset all scores |

---

## 🧠 Minimax Algorithm Explanation

Minimax is a **decision-making algorithm** used in two-player zero-sum games. It builds a complete game tree from the current position and evaluates every possible future state.

### Evaluation Scores

| Outcome | Score |
|:--------|:-----:|
| AI Wins | `+1` |
| Human Wins | `-1` |
| Draw | `0` |

### Algorithm Rules
- **AI is the Maximising player** — always picks the move with the **highest** score
- **Human is the Minimising player** — AI assumes optimal human play (lowest score)
- Recursion continues until every branch reaches a **terminal state** (win/draw)
- The root call returns the move with the **best worst-case outcome** for the AI

### Code Snippet

```python
def minimax(board, is_maximising):
    winner, _ = _check_winner_pure(board)
    if winner == AI:    return +1
    if winner == HUMAN: return -1
    if _check_draw_pure(board): return 0

    if is_maximising:
        best = -2
        for i in range(9):
            if board[i] == EMPTY:
                board[i] = AI
                best = max(best, minimax(board, False))
                board[i] = EMPTY
        return best
    else:
        best = +2
        for i in range(9):
            if board[i] == EMPTY:
                board[i] = HUMAN
                best = min(best, minimax(board, True))
                board[i] = EMPTY
        return best
```

> **Why is the AI unbeatable?**
> Minimax evaluates *every* possible game continuation. With perfect play, Tic-Tac-Toe is always a draw — but any human mistake is immediately capitalised on.

---

## 🏗️ Architecture & Code Organisation

The project is a single well-structured Python file with a clean separation of concerns:

### Pure Game Logic (module-level functions)
| Function | Purpose |
|:---------|:--------|
| `_check_winner_pure(board)` | Scan all 8 win lines — no GUI side-effects |
| `_check_draw_pure(board)` | Check if board is full |
| `minimax(board, is_maximising)` | Recursive game tree search |
| `get_best_move(board)` | Return optimal cell index for the AI |

### GUI Methods (TicTacToeApp class)
| Method | Purpose |
|:-------|:--------|
| `_configure_window()` | Maximise window on startup |
| `_build_fonts()` | Create fixed-size font objects |
| `_build_ui()` | Assemble all UI sections via grid layout |
| `_build_header()` | Title banner and subtitle |
| `_build_scoreboard()` | Three score cards (YOU / DRAW / AI) |
| `_build_status_bar()` | Status message + Play Again button |
| `_build_board()` | 3×3 cell grid with weight-based sizing |
| `_build_controls()` | Restart / New Match buttons |
| `_build_footer()` | Developer credit |
| `on_cell_click(idx)` | Handle human move with validation |
| `ai_move()` | Execute AI move after thinking delay |
| `_check_and_handle_outcome()` | Detect win/draw after each move |
| `_place_marker(idx, player)` | Update board state and button |
| `check_winner()` / `check_draw()` | Win/draw detection |
| `_handle_win(winner)` | Update scores, highlight, show hint |
| `_handle_draw()` | Update scores, show hint |
| `_highlight_winner()` | Golden highlight on winning triple |
| `reset_board()` | Clear board, keep scores |
| `new_match()` | Clear board and scores |
| `update_status(msg, color)` | Update status label |
| `update_scoreboard()` | Refresh score counters |

### Layout Design
The root window uses Tkinter's **grid geometry manager** with `weight=1` on the board row, ensuring it always fills available vertical space. Inside the board, all 3 rows and 3 columns have `weight=1` with `minsize` guards so all **9 cells are always visible** regardless of window size or OS timing.

---

## 📁 Project Structure

```
Codsoft - TicTacToe/
│
├── tictactoe.py          ← Complete application (single file, zero dependencies)
├── README.md             ← This file
├── .gitignore            ← Git ignore rules
│
├── screenshots/          ← UI screenshots
│   └── README.md         ← Screenshot guide
│
└── assets/               ← Additional assets
    └── README.md         ← Assets guide
```

---

## 📸 Screenshots

> Run the app, take screenshots with `Win + Shift + S`, and place them in `screenshots/`.

| Welcome Screen | Gameplay | AI Wins | Draw |
|:-:|:-:|:-:|:-:|
| *(Add screenshot)* | *(Add screenshot)* | *(Add screenshot)* | *(Add screenshot)* |

---

## ⚙️ Installation

### Prerequisites
- **Python 3.8 or higher** — [Download here](https://www.python.org/downloads/)
- `tkinter` is **included with Python** by default — no pip installs required

### Clone the Repository

```bash
git clone https://github.com/your-username/Codsoft-TicTacToe.git
cd "Codsoft - TicTacToe"
```

---

## ▶️ How To Run

```bash
python tictactoe.py
```

The game window opens immediately, **maximised** and ready to play.

### Troubleshooting

| Issue | Fix |
|:------|:----|
| `ModuleNotFoundError: No module named 'tkinter'` | Linux only: `sudo apt-get install python3-tk` |
| Window doesn't open | Ensure Python ≥ 3.8 is installed and on PATH |
| Board appears small | Maximise the window — the board auto-scales |

---

## 🎨 Colour Palette

| Role | Colour | Hex |
|:-----|:-------|:----|
| Background | ■ Deep Black | `#121212` |
| Board / Cells | ■ Dark Grey | `#1E1E1E` |
| Control Buttons | ■ Medium Grey | `#2D2D2D` |
| X Symbol / Accent | ■ Green | `#4CAF50` |
| O Symbol / AI | ■ Red | `#F44336` |
| Winning Highlight | ■ Golden Yellow | `#FFD54F` |
| Header Background | ■ Navy Dark | `#1A1A2E` |
| Primary Text | ■ White | `#FFFFFF` |
| Secondary Text | ■ Light Grey | `#B0B0B0` |
| Status Bar | ■ Near Black | `#0D0D0D` |

---

## 📊 Algorithm Complexity

| Metric | Value |
|:-------|:------|
| **Total possible games** | 255,168 |
| **Terminal states** | 138 distinct end positions |
| **Time Complexity** | O(b^d) — b=9 branches, d=9 depth |
| **Space Complexity** | O(d) — call stack only |
| **Pruning** | None required (search space is tiny) |

---

## 🚀 Future Enhancements

| Enhancement | Description |
|:------------|:------------|
| 🎯 **Difficulty Levels** | Easy (random moves), Medium (partial search), Hard (full minimax) |
| ⚡ **Alpha-Beta Pruning** | Prune branches that cannot affect the outcome — faster for larger boards |
| 🌐 **Online Multiplayer** | Real-time Human vs Human over a network using Python sockets |
| 📐 **Larger Boards** | 4×4 or 5×5 boards with extended win conditions |
| 🎵 **Sound Effects** | Move click, win fanfare, draw jingle |
| 🏆 **Persistent Leaderboard** | Save session scores to a local JSON file |
| 🌙 **Theme Switcher** | Toggle between Dark Mode and Light Mode |
| 🤖 **AI vs AI Mode** | Watch two Minimax agents play against each other |

---

## 🏅 Code Quality

- ✅ Single-file — runs immediately with `python tictactoe.py`
- ✅ Zero external dependencies (standard library only)
- ✅ Pure game logic functions (no GUI side-effects in minimax)
- ✅ Grid-based responsive layout — 3×3 board always visible
- ✅ Full session score tracking
- ✅ Play-again flow with button and keyboard shortcut (`R`)
- ✅ Winning cells highlighted in gold
- ✅ Hover effects and dark theme throughout
- ✅ Fully documented with docstrings and inline comments

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

---

<div align="center">

**Made with ❤️ by Sanjeevikumar D**
*CodSoft Artificial Intelligence Internship — Task 2*

⭐ *If you found this helpful, please star the repository!*

</div>
