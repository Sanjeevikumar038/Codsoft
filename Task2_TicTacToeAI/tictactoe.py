"""
============================================================
  Tic-Tac-Toe AI  —  Modern Neon UI
  ============================================================
  Developer  : Sanjeevikumar D
  Internship : CodSoft Artificial Intelligence Internship
  Task       : Task 2 – Tic-Tac-Toe AI
  ============================================================
  Run: python tictactoe.py
  ============================================================
"""

import sys
import math
import time
import tkinter as tk
from tkinter import font as tkfont

# ──────────────────────────────────────────────────────────
#  NEON CYBERPUNK PALETTE
# ──────────────────────────────────────────────────────────
BG          = "#07070F"   # ultra-dark background
PANEL       = "#0E0E1C"   # card / panel background
CELL_BG     = "#12122A"   # empty cell
CELL_HOVER  = "#1A1A3A"   # cell hover
CELL_BORDER = "#252550"   # cell border (idle)
GRID_LINE   = "#1C1C3C"   # thin grid separator

X_COLOR     = "#00F5FF"   # neon cyan  (human)
X_GLOW_DARK = "#001A20"

O_COLOR     = "#FF2D78"   # neon pink  (AI)
O_GLOW_DARK = "#1A0010"

WIN_BRIGHT  = "#FFD700"   # gold       (winner cell – bright)
WIN_DIM     = "#7A6500"   # gold dim   (winner pulse)

ACCENT      = "#7C3AED"   # purple accent
ACCENT_DIM  = "#3D1D78"

TEXT        = "#E2E8F0"
SUBTEXT     = "#64748B"
HEADER_BG   = "#09091A"

WINDOW_WIDTH  = 900
WINDOW_HEIGHT = 700

HUMAN = "X"
AI    = "O"
EMPTY = ""

WIN_LINES = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),   # rows
    (0, 3, 6), (1, 4, 7), (2, 5, 8),   # columns
    (0, 4, 8), (2, 4, 6),              # diagonals
]

# ──────────────────────────────────────────────────────────
#  PURE GAME-LOGIC  (no GUI side-effects)
# ──────────────────────────────────────────────────────────

def _check_winner_pure(board):
    for a, b, c in WIN_LINES:
        if board[a] != EMPTY and board[a] == board[b] == board[c]:
            return board[a], (a, b, c)
    return None, None


def _check_draw_pure(board):
    return all(cell != EMPTY for cell in board)


def minimax(board, is_maximising):
    winner, _ = _check_winner_pure(board)
    if winner == AI:    return 1
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
        best = 2
        for i in range(9):
            if board[i] == EMPTY:
                board[i] = HUMAN
                best = min(best, minimax(board, True))
                board[i] = EMPTY
        return best


def get_best_move(board):
    best_score, best_idx = -2, None
    for i in range(9):
        if board[i] == EMPTY:
            board[i] = AI
            score = minimax(board, False)
            board[i] = EMPTY
            if score > best_score:
                best_score, best_idx = score, i
    return best_idx


# ──────────────────────────────────────────────────────────
#  COLOUR HELPERS
# ──────────────────────────────────────────────────────────

def _hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def _rgb_to_hex(r, g, b):
    return f"#{int(r):02X}{int(g):02X}{int(b):02X}"


def _lerp_color(c1, c2, t):
    """Linear interpolation between two hex colours."""
    r1, g1, b1 = _hex_to_rgb(c1)
    r2, g2, b2 = _hex_to_rgb(c2)
    return _rgb_to_hex(r1 + (r2-r1)*t, g1 + (g2-g1)*t, b1 + (b2-b1)*t)


def _ease_out_cubic(t):
    return 1 - (1 - t) ** 3


# ──────────────────────────────────────────────────────────
#  MAIN APPLICATION
# ──────────────────────────────────────────────────────────

class TicTacToeApp:
    """
    Neon Cyberpunk Tic-Tac-Toe with full canvas animation.

    Animation system
    ----------------
    A single `_loop()` method runs at ~60 fps via `root.after(16, ...)`.
    Each frame it:
      1. Advances per-cell draw animations (progress 0→1)
      2. Advances the win-pulse oscillator
      3. Advances the title shimmer
      4. Advances AI thinking dots
      5. Redraws the board canvas

    Cell draw animations
    --------------------
    When a marker is placed, `cell_draw_progress[idx]` is set to 0 and
    `cell_draw_start[idx]` records `time.time()`.  Each frame we compute
    elapsed / DRAW_DURATION, apply an ease-out-cubic, and pass the result
    to `_draw_x()` or `_draw_o()`.  Both functions draw layered neon lines
    /arcs to simulate a glow effect without requiring alpha transparency.
    """

    DRAW_DURATION = 0.35   # seconds for X / O stroke animation
    GAP_RATIO     = 0.03   # cell gap as fraction of board size
    CELL_RADIUS   = 18     # px corner radius for cell rounded rect
    THINK_PERIOD  = 400    # ms per thinking-dot frame

    def __init__(self, root: tk.Tk):
        self.root = root
        self._configure_window()

        # ── Scores ──
        self.human_wins = 0
        self.ai_wins    = 0
        self.draws      = 0

        # ── Game state ──
        self.board        = [EMPTY] * 9
        self.game_over    = False
        self.current_turn = HUMAN
        self.winner_cells = []

        # ── Animation state ──
        self.cell_draw_progress = [1.0] * 9  # 1.0 = fully drawn / empty
        self.cell_draw_start    = [0.0]  * 9
        self.hover_cell         = -1
        self.win_pulse          = 0.0    # 0→1 oscillator
        self.win_pulse_dir      = 1
        self.title_t            = 0.0    # title shimmer phase
        self.think_dots         = 0      # 0,1,2 rotating dot index
        self.think_last         = 0      # ms timestamp of last dot advance
        self.status_text        = ""
        self.status_color       = TEXT
        self.cells              = []     # [(x1,y1,x2,y2) * 9]

        self._build_fonts()
        self._build_ui()
        self._loop()
        self.root.after(300, self._show_welcome)

    # ── Window ──────────────────────────────────────────────
    def _configure_window(self):
        self.root.title("Tic-Tac-Toe AI  |  CodSoft AI Internship")
        self.root.configure(bg=BG)
        self.root.minsize(WINDOW_WIDTH, WINDOW_HEIGHT)
        if sys.platform.startswith("win"):
            self.root.state("zoomed")
        else:
            sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
            self.root.geometry(f"{sw}x{sh}+0+0")

    # ── Fonts ────────────────────────────────────────────────
    def _build_fonts(self):
        self.fnt_title  = tkfont.Font(family="Segoe UI", size=22, weight="bold")
        self.fnt_sub    = tkfont.Font(family="Segoe UI", size=9)
        self.fnt_score  = tkfont.Font(family="Segoe UI", size=16, weight="bold")
        self.fnt_slabel = tkfont.Font(family="Segoe UI", size=10, weight="bold")
        self.fnt_status = tkfont.Font(family="Segoe UI", size=14, weight="bold")
        self.fnt_btn    = tkfont.Font(family="Segoe UI", size=11, weight="bold")

    # ─────────────────────────────────────────────────────────
    #  UI CONSTRUCTION
    # ─────────────────────────────────────────────────────────

    def _build_ui(self):
        # Root grid: column 0 fills width; row 3 (board) takes spare height
        self.root.grid_columnconfigure(0, weight=1)
        for r in range(6):
            self.root.grid_rowconfigure(r, weight=(1 if r == 3 else 0))

        self._build_header()      # row 0
        self._build_scoreboard()  # row 1
        self._build_status()      # row 2
        self._build_board()       # row 3
        self._build_controls()    # row 4
        self._build_footer()      # row 5

        self.root.bind("<r>", lambda e: self.reset_board())
        self.root.bind("<R>", lambda e: self.reset_board())

    # ── Header ──────────────────────────────────────────────
    def _build_header(self):
        hf = tk.Frame(self.root, bg=HEADER_BG, pady=14)
        hf.grid(row=0, column=0, sticky="ew")

        # Animated title via Canvas label
        self.title_canvas = tk.Canvas(hf, bg=HEADER_BG, bd=0,
                                      highlightthickness=0, height=36)
        self.title_canvas.pack()
        self._title_text_id = self.title_canvas.create_text(
            0, 18, text="🎯  TIC-TAC-TOE  AI",
            font=self.fnt_title, fill=X_COLOR, anchor="center")
        # Resize title canvas to fit text
        hf.update_idletasks()
        bbox = self.title_canvas.bbox(self._title_text_id)
        if bbox:
            w = bbox[2] - bbox[0] + 20
            self.title_canvas.configure(width=w)
            self.title_canvas.coords(self._title_text_id, w//2, 18)



        # Neon accent bar
        bar = tk.Canvas(self.root, bg=BG, height=3,
                         bd=0, highlightthickness=0)
        bar.grid(row=0, column=0, sticky="sew")
        bar.bind("<Configure>",
                 lambda e: (bar.delete("all"),
                             bar.create_rectangle(0, 0, e.width, 3,
                                                  fill=ACCENT, outline="")))

    # ── Scoreboard ──────────────────────────────────────────
    def _build_scoreboard(self):
        sf = tk.Frame(self.root, bg=BG, pady=10)
        sf.grid(row=1, column=0, sticky="ew", padx=40)
        sf.grid_columnconfigure((0, 1, 2), weight=1)

        cards = [
            ("YOU", "X", X_COLOR,   ACCENT_DIM, "human"),
            ("DRAW","", "#9CA3AF",  "#1F2937",  "draw"),
            ("AI",  "O", O_COLOR,   "#1A0818",  "ai"),
        ]
        for col, (title, sym, color, bg2, key) in enumerate(cards):
            card = tk.Frame(sf, bg=PANEL, padx=24, pady=10)
            card.grid(row=0, column=col, sticky="ew", padx=8)

            top = tk.Frame(card, bg=PANEL)
            top.pack()
            if sym:
                tk.Label(top, text=sym, font=tkfont.Font(family="Segoe UI",
                         size=13, weight="bold"),
                         bg=PANEL, fg=color).pack(side=tk.LEFT, padx=(0, 6))
            tk.Label(top, text=title, font=self.fnt_slabel,
                     bg=PANEL, fg=SUBTEXT).pack(side=tk.LEFT)

            var = tk.StringVar(value="0")
            setattr(self, f"score_{key}_var", var)
            tk.Label(card, textvariable=var, font=self.fnt_score,
                     bg=PANEL, fg=color).pack()

            # Coloured bottom border
            tk.Frame(card, height=3, bg=color).pack(fill=tk.X, pady=(6, 0))

    # ── Status Bar ──────────────────────────────────────────
    def _build_status(self):
        sf = tk.Frame(self.root, bg=PANEL, pady=8)
        sf.grid(row=2, column=0, sticky="ew", padx=40, pady=(0, 4))

        self.status_var = tk.StringVar(value="")
        self.status_lbl = tk.Label(sf, textvariable=self.status_var,
                                   font=self.fnt_status, bg=PANEL,
                                   fg=TEXT, pady=2)
        self.status_lbl.pack()

        # Play-again button (hidden until game over)
        self.play_again_btn = tk.Button(
            sf,
            text="▶  Play Again?   (or press  R)",
            font=tkfont.Font(family="Segoe UI", size=11, weight="bold"),
            bg=X_COLOR, fg=BG,
            activebackground="#00C8D0", activeforeground=BG,
            padx=18, pady=6, relief=tk.FLAT, cursor="hand2",
            command=self.reset_board)

    # ── Board Canvas ─────────────────────────────────────────
    def _build_board(self):
        """
        A single Canvas widget occupies the board row.
        All drawing is done imperatively in _draw_board().
        Click / hover events are resolved by comparing mouse
        coordinates against self.cells[] bounding boxes.
        """
        self.board_canvas = tk.Canvas(
            self.root, bg=BG, bd=0, highlightthickness=0, cursor="hand2")
        self.board_canvas.grid(row=3, column=0, sticky="nsew")

        self.board_canvas.bind("<Configure>", lambda e: None)  # redraw via loop
        self.board_canvas.bind("<Button-1>",  self._on_canvas_click)
        self.board_canvas.bind("<Motion>",    self._on_canvas_hover)
        self.board_canvas.bind("<Leave>",     self._on_canvas_leave)

    # ── Controls ─────────────────────────────────────────────
    def _build_controls(self):
        cf = tk.Frame(self.root, bg=BG, pady=10)
        cf.grid(row=4, column=0, sticky="ew")
        inner = tk.Frame(cf, bg=BG)
        inner.pack()

        def neon_btn(parent, text, cmd, hover_bg, active_fg=BG):
            b = tk.Button(parent, text=text, font=self.fnt_btn,
                          bg="#1A1A30", fg=TEXT,
                          activebackground=hover_bg,
                          activeforeground=active_fg,
                          padx=24, pady=10, relief=tk.FLAT,
                          cursor="hand2", bd=0, command=cmd)
            b.bind("<Enter>", lambda e: b.configure(bg=hover_bg, fg=active_fg))
            b.bind("<Leave>", lambda e: b.configure(bg="#1A1A30", fg=TEXT))
            return b

        r = neon_btn(inner, "🔄  Restart Game", self.reset_board, ACCENT)
        r.pack(side=tk.LEFT, padx=12)

        n = neon_btn(inner, "🆕  New Match", self.new_match, O_COLOR)
        n.pack(side=tk.LEFT, padx=12)

    # ── Footer ───────────────────────────────────────────────
    def _build_footer(self):
        ff = tk.Frame(self.root, bg=BG)
        ff.grid(row=5, column=0, sticky="ew")
        tk.Frame(ff, height=1, bg=CELL_BORDER).pack(fill=tk.X)
        tk.Label(ff, text="Developed by Sanjeevikumar D  ·  CodSoft Artificial Intelligence Internship",
                 font=self.fnt_sub, bg=BG, fg=SUBTEXT).pack(pady=5)

    # ─────────────────────────────────────────────────────────
    #  ANIMATION LOOP  (~60 fps)
    # ─────────────────────────────────────────────────────────

    def _loop(self):
        now = time.time()

        # Advance cell draw animations
        for i in range(9):
            if self.board[i] != EMPTY and self.cell_draw_progress[i] < 1.0:
                elapsed = now - self.cell_draw_start[i]
                self.cell_draw_progress[i] = min(1.0, elapsed / self.DRAW_DURATION)

        # Win pulse oscillator
        if self.game_over and self.winner_cells:
            self.win_pulse += 0.04 * self.win_pulse_dir
            if self.win_pulse >= 1.0:
                self.win_pulse_dir = -1
            elif self.win_pulse <= 0.0:
                self.win_pulse_dir = 1

        # Title shimmer
        self.title_t = (self.title_t + 0.02) % (2 * math.pi)
        shimmer = 0.5 + 0.5 * math.sin(self.title_t)
        title_color = _lerp_color(X_COLOR, ACCENT, shimmer)
        self.title_canvas.itemconfig(self._title_text_id, fill=title_color)

        # Thinking dots (only while AI's turn is pending)
        if not self.game_over and self.current_turn == AI:
            ms_now = int(time.time() * 1000)
            if ms_now - self.think_last > self.THINK_PERIOD:
                self.think_dots = (self.think_dots + 1) % 3
                self.think_last = ms_now
                dots = ["●  ○  ○", "○  ●  ○", "○  ○  ●"][self.think_dots]
                self._set_status(f"AI Thinking  {dots}", "#FF9800")

        # Redraw the board
        self._draw_board()

        self.root.after(16, self._loop)

    # ─────────────────────────────────────────────────────────
    #  BOARD DRAWING
    # ─────────────────────────────────────────────────────────

    def _compute_cells(self):
        """Compute the 9 cell bounding boxes centred in the canvas."""
        c  = self.board_canvas
        cw = c.winfo_width()
        ch = c.winfo_height()
        if cw < 10 or ch < 10:
            return False

        board_size = min(cw, ch) * 0.88
        gap        = board_size * self.GAP_RATIO + 4
        cell_size  = (board_size - 2 * gap) / 3

        ox = (cw - board_size) / 2
        oy = (ch - board_size) / 2

        self.cells = []
        for row in range(3):
            for col in range(3):
                x1 = ox + col * (cell_size + gap)
                y1 = oy + row * (cell_size + gap)
                self.cells.append((x1, y1, x1 + cell_size, y1 + cell_size))
        return True

    def _draw_board(self):
        """Full board redraw — called every animation frame."""
        cv = self.board_canvas
        cv.delete("all")

        if not self._compute_cells():
            return

        # ── Board ambient glow border ──
        if self.cells:
            x1 = self.cells[0][0] - 8
            y1 = self.cells[0][1] - 8
            x2 = self.cells[8][2] + 8
            y2 = self.cells[8][3] + 8
            self._round_rect(cv, x1-4, y1-4, x2+4, y2+4, 24,
                             fill="", outline=ACCENT_DIM, width=2)
            self._round_rect(cv, x1,   y1,   x2,   y2,   20,
                             fill=GRID_LINE, outline=ACCENT, width=1)

        # ── Cells ──
        for i in range(9):
            self._draw_cell(i)

    def _draw_cell(self, idx):
        cv = self.board_canvas
        x1, y1, x2, y2 = self.cells[idx]
        is_winner = idx in self.winner_cells
        is_hover  = idx == self.hover_cell
        marker    = self.board[idx]

        # ── Background & border ──
        if is_winner:
            t   = self.win_pulse
            bg  = _lerp_color("#1A1400", "#2A2000", t)
            bdr = _lerp_color(WIN_DIM, WIN_BRIGHT, t)
            bw  = 3
        elif is_hover and not self.game_over and marker == EMPTY:
            bg  = CELL_HOVER
            bdr = _lerp_color(CELL_BORDER, ACCENT, 0.6)
            bw  = 2
        else:
            bg  = CELL_BG
            bdr = CELL_BORDER
            bw  = 1

        self._round_rect(cv, x1, y1, x2, y2,
                         self.CELL_RADIUS, fill=bg, outline=bdr, width=bw)

        # ── Marker ──
        if marker != EMPTY:
            p = _ease_out_cubic(self.cell_draw_progress[idx])
            if marker == HUMAN:
                self._draw_x(idx, p)
            else:
                self._draw_o(idx, p)

    def _draw_x(self, idx, progress):
        """Draw neon cyan X with layered glow."""
        cv = self.board_canvas
        x1, y1, x2, y2 = self.cells[idx]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        sz = min(x2-x1, y2-y1) * 0.27 * progress

        # Glow layers: (width, colour) from outermost (darkest/widest) to core
        layers = [
            (18, "#001A20"), (13, "#003340"),
            (8,  "#006080"), (4,  "#00C8D0"), (2, X_COLOR),
        ]
        for w, col in layers:
            cv.create_line(cx-sz, cy-sz, cx+sz, cy+sz,
                           width=w, fill=col, capstyle="round")
            cv.create_line(cx+sz, cy-sz, cx-sz, cy+sz,
                           width=w, fill=col, capstyle="round")

    def _draw_o(self, idx, progress):
        """
        Draw neon pink O with layered glow.

        Tkinter bug workaround
        ----------------------
        create_arc with extent=0 or extent=±360 renders NOTHING.
        At full progress (1.0) we use create_oval instead, which is
        always visible.  During animation we clamp extent to -359.9
        so it never hits the degenerate boundary.
        """
        cv = self.board_canvas
        x1, y1, x2, y2 = self.cells[idx]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        r = min(x2-x1, y2-y1) * 0.30

        layers = [
            (18, "#1A0010"), (13, "#33001F"),
            (8,  "#660040"), (4,  "#CC0065"), (2, O_COLOR),
        ]

        if progress >= 0.999:
            # Fully drawn — use create_oval (immune to the ±360 arc bug)
            for w, col in layers:
                cv.create_oval(cx-r, cy-r, cx+r, cy+r,
                               outline=col, width=w, fill="")
        else:
            # Animating — clockwise arc; clamp away from exactly -360
            ext = max(-359.9, -360.0 * progress)
            for w, col in layers:
                cv.create_arc(cx-r, cy-r, cx+r, cy+r,
                              start=90, extent=ext,
                              style="arc", outline=col, width=w)


    # ── Rounded rectangle helper ─────────────────────────────
    @staticmethod
    def _round_rect(canvas, x1, y1, x2, y2, r, **kw):
        """Draw a rounded rectangle on a Canvas widget."""
        pts = [
            x1+r, y1,   x2-r, y1,
            x2,   y1,   x2,   y1+r,
            x2,   y2-r, x2,   y2,
            x2-r, y2,   x1+r, y2,
            x1,   y2,   x1,   y2-r,
            x1,   y1+r, x1,   y1,
        ]
        return canvas.create_polygon(pts, smooth=True, **kw)

    # ─────────────────────────────────────────────────────────
    #  CANVAS INTERACTION
    # ─────────────────────────────────────────────────────────

    def _cell_at(self, x, y):
        """Return the cell index under (x,y), or -1."""
        for i, (x1, y1, x2, y2) in enumerate(self.cells):
            if x1 <= x <= x2 and y1 <= y <= y2:
                return i
        return -1

    def _on_canvas_click(self, event):
        idx = self._cell_at(event.x, event.y)
        if idx >= 0:
            self.on_cell_click(idx)

    def _on_canvas_hover(self, event):
        idx = self._cell_at(event.x, event.y)
        self.hover_cell = idx

    def _on_canvas_leave(self, event):
        self.hover_cell = -1

    # ─────────────────────────────────────────────────────────
    #  GAME LOGIC  (identical core logic, modern presentation)
    # ─────────────────────────────────────────────────────────

    def on_cell_click(self, idx):
        if self.game_over:          return
        if self.current_turn != HUMAN: return
        if self.board[idx] != EMPTY:   return

        self._place_marker(idx, HUMAN)
        if self._check_outcome():   return

        self.current_turn = AI
        self.think_dots   = 0
        self.think_last   = int(time.time() * 1000)
        self._set_status("AI Thinking  ●  ○  ○", "#FF9800")
        self.root.after(900, self.ai_move)

    def ai_move(self):
        best = get_best_move(self.board)
        if best is not None:
            self._place_marker(best, AI)

        if self._check_outcome():   return

        self.current_turn = HUMAN
        self._set_status("Your Turn  ✏️", X_COLOR)

    def _check_outcome(self):
        winner, cells = _check_winner_pure(self.board)
        if winner:
            self.winner_cells = list(cells)
            self._handle_win(winner)
            return True
        if _check_draw_pure(self.board):
            self._handle_draw()
            return True
        return False

    def _place_marker(self, idx, player):
        self.board[idx] = player
        # Trigger draw animation
        self.cell_draw_progress[idx] = 0.0
        self.cell_draw_start[idx]    = time.time()

    def _handle_win(self, winner):
        self.game_over = True
        self.win_pulse = 0.0
        if winner == HUMAN:
            self.human_wins += 1
            self._set_status("🎉  You Win!  Congratulations!", X_COLOR)
        else:
            self.ai_wins += 1
            self._set_status("🤖  AI Wins!  Better luck next time.", O_COLOR)
        self.update_scoreboard()
        self._show_play_again()

    def _handle_draw(self):
        self.game_over = True
        self.draws += 1
        self._set_status("🤝  It's a Draw!  Well played.", "#9CA3AF")
        self.update_scoreboard()
        self._show_play_again()

    def _show_play_again(self):
        self.play_again_btn.pack(pady=(4, 2))

    def _hide_play_again(self):
        self.play_again_btn.pack_forget()

    # ─────────────────────────────────────────────────────────
    #  STATUS & SCOREBOARD
    # ─────────────────────────────────────────────────────────

    def _set_status(self, msg, color=TEXT):
        self.status_var.set(msg)
        self.status_lbl.configure(fg=color)

    def update_scoreboard(self):
        self.score_human_var.set(str(self.human_wins))
        self.score_draw_var.set(str(self.draws))
        self.score_ai_var.set(str(self.ai_wins))

    # ─────────────────────────────────────────────────────────
    #  RESET
    # ─────────────────────────────────────────────────────────

    def reset_board(self):
        self.board               = [EMPTY] * 9
        self.game_over           = False
        self.current_turn        = HUMAN
        self.winner_cells        = []
        self.win_pulse           = 0.0
        self.win_pulse_dir       = 1
        self.cell_draw_progress  = [1.0] * 9
        self._hide_play_again()
        self._set_status("Your Turn  ✏️", X_COLOR)

    def new_match(self):
        self.human_wins = self.ai_wins = self.draws = 0
        self.update_scoreboard()
        self.reset_board()
        self._set_status("New Match!  Your Turn  ✏️", X_COLOR)

    # ─────────────────────────────────────────────────────────
    #  WELCOME OVERLAY
    # ─────────────────────────────────────────────────────────

    def _show_welcome(self):
        ov = tk.Toplevel(self.root)
        ov.title("")
        ov.configure(bg=PANEL)
        ov.resizable(False, False)
        ov.transient(self.root)
        ov.grab_set()

        ow, oh = 500, 340
        self.root.update_idletasks()
        rx = self.root.winfo_rootx()
        ry = self.root.winfo_rooty()
        rw = self.root.winfo_width()
        rh = self.root.winfo_height()
        ov.geometry(f"{ow}x{oh}+{rx+(rw-ow)//2}+{ry+(rh-oh)//2}")

        # Neon top bar
        tk.Frame(ov, height=3, bg=X_COLOR).pack(fill=tk.X)

        tk.Label(ov, text="🎯  Welcome to Tic-Tac-Toe AI",
                 font=tkfont.Font(family="Segoe UI", size=17, weight="bold"),
                 bg=PANEL, fg=X_COLOR).pack(pady=(20, 6))

        tk.Label(ov,
                 text="Challenge an unbeatable AI\npowered by the Minimax Algorithm.",
                 font=tkfont.Font(family="Segoe UI", size=12),
                 bg=PANEL, fg=TEXT, justify=tk.CENTER).pack()

        tk.Frame(ov, height=1, bg=CELL_BORDER).pack(
            fill=tk.X, padx=30, pady=14)

        rows = [
            (f"  {HUMAN}   You play as X   — Neon Cyan",   X_COLOR),
            (f"  {AI}   AI plays as O  — Neon Pink",  O_COLOR),
            ("  👤  You always move first",            TEXT),
            ("  ⚡  AI uses Minimax — it never loses!",ACCENT),
        ]
        for txt, col in rows:
            tk.Label(ov, text=txt,
                     font=tkfont.Font(family="Segoe UI", size=11),
                     bg=PANEL, fg=col, anchor="w").pack(
                         fill=tk.X, padx=40, pady=1)

        def close():
            ov.destroy()
            self._set_status("Your Turn  ✏️", X_COLOR)

        btn = tk.Button(ov, text="🚀  Let's Play!",
                        font=tkfont.Font(family="Segoe UI", size=13, weight="bold"),
                        bg=ACCENT, fg=TEXT,
                        activebackground=X_COLOR, activeforeground=BG,
                        padx=32, pady=10, relief=tk.FLAT,
                        cursor="hand2", command=close)
        btn.pack(pady=18)
        btn.bind("<Enter>", lambda e: btn.configure(bg=X_COLOR, fg=BG))
        btn.bind("<Leave>", lambda e: btn.configure(bg=ACCENT, fg=TEXT))

        # Neon bottom bar
        tk.Frame(ov, height=3, bg=O_COLOR).pack(
            fill=tk.X, side=tk.BOTTOM)


# ──────────────────────────────────────────────────────────
#  ENTRY POINT
# ──────────────────────────────────────────────────────────

def main():
    root = tk.Tk()
    app  = TicTacToeApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
