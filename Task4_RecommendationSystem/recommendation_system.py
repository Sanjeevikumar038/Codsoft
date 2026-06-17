"""
╔═══════════════════════════════════════════════════════════════════╗
║          CineMind AI - Intelligent Movie Recommendation System     ║
║          CodSoft AI Internship - Task 4                           ║
║          Developer: Sanjeevikumar D                               ║
╚═══════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import json
import random
import threading
import datetime
import tkinter as tk
from tkinter import messagebox, filedialog
from typing import Optional, List, Dict, Tuple
import math

# ─── Third-party imports ────────────────────────────────────────────────────
try:
    import customtkinter as ctk
except ImportError:
    print("Installing CustomTkinter...")
    os.system(f"{sys.executable} -m pip install customtkinter --quiet")
    import customtkinter as ctk

try:
    import pandas as pd
except ImportError:
    os.system(f"{sys.executable} -m pip install pandas --quiet")
    import pandas as pd

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
except ImportError:
    os.system(f"{sys.executable} -m pip install scikit-learn numpy --quiet")
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS & THEME
# ═══════════════════════════════════════════════════════════════════════════════

COLORS = {
    "bg":           "#0F1117",
    "sidebar":      "#13151F",
    "card":         "#1A1D29",
    "card_hover":   "#20243A",
    "accent":       "#6C63FF",
    "accent2":      "#8B82FF",
    "accent_glow":  "#2A275C",
    "success":      "#4CAF50",
    "warning":      "#FFC107",
    "danger":       "#FF5252",
    "text":         "#FFFFFF",
    "text_sec":     "#A0A0B0",
    "text_muted":   "#606080",
    "border":       "#252840",
    "input_bg":     "#0D0F1A",
    "gradient1":    "#6C63FF",
    "gradient2":    "#FF6584",
    "tag_sf":       "#6C63FF",
    "tag_action":   "#FF6584",
    "tag_drama":    "#4CAF50",
    "tag_horror":   "#FF5252",
    "tag_romance":  "#FF69B4",
    "tag_comedy":   "#FFC107",
    "tag_thriller": "#FF8C00",
    "tag_fantasy":  "#9B59B6",
    "tag_hist":     "#795548",
    "tag_default":  "#607D8B",
}

WINDOW_WIDTH  = 1400
WINDOW_HEIGHT = 850
APP_TITLE     = "CineMind AI – Smart Recommendation Engine"
DATASET_PATH  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "movies.csv")
FAVORITES_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "favorites.json")
HISTORY_FILE   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "history.json")

GENRE_COLORS = {
    "science fiction": COLORS["tag_sf"],
    "action":          COLORS["tag_action"],
    "drama":           COLORS["tag_drama"],
    "horror":          COLORS["tag_horror"],
    "romance":         COLORS["tag_romance"],
    "comedy":          COLORS["tag_comedy"],
    "thriller":        COLORS["tag_thriller"],
    "fantasy":         COLORS["tag_fantasy"],
    "historical":      COLORS["tag_hist"],
    "animation":       "#00BCD4",
    "adventure":       "#FF9800",
    "musical":         "#E91E63",
    "biographical":    "#795548",
    "crime":           "#F44336",
    "mystery":         "#9C27B0",
    "war":             "#607D8B",
    "superhero":       "#FF5722",
}


def get_genre_color(genre: str) -> str:
    """Return a color code for the given genre string."""
    g = genre.lower().strip()
    for key, color in GENRE_COLORS.items():
        if key in g:
            return color
    return COLORS["tag_default"]


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LAYER
# ═══════════════════════════════════════════════════════════════════════════════

class MovieDataset:
    """Handles loading, validation, and management of the movie dataset."""

    REQUIRED_COLUMNS: List[str] = ["movie_id", "title", "genre", "keywords", "description"]

    def __init__(self, path: str = DATASET_PATH) -> None:
        self.path = path
        self.df: pd.DataFrame = pd.DataFrame()
        self._load()

    # ── Public API ────────────────────────────────────────────────────────────

    def reload(self) -> None:
        """Reload the dataset from disk."""
        self._load()

    def get_all_titles(self) -> List[str]:
        return self.df["title"].tolist()

    def get_movie_by_title(self, title: str) -> Optional[Dict]:
        row = self.df[self.df["title"].str.lower() == title.lower()]
        if row.empty:
            return None
        return row.iloc[0].to_dict()

    def search_titles(self, query: str) -> List[str]:
        if not query.strip():
            return self.get_all_titles()
        q = query.lower()
        mask = self.df["title"].str.lower().str.contains(q, na=False)
        return self.df[mask]["title"].tolist()

    def get_random_movie(self) -> Optional[Dict]:
        if self.df.empty:
            return None
        return self.df.sample(1).iloc[0].to_dict()

    def total_movies(self) -> int:
        return len(self.df)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _load(self) -> None:
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Dataset not found: {self.path}")
        try:
            df = pd.read_csv(self.path)
            self._validate(df)
            self.df = df.dropna(subset=["title", "genre", "description"]).copy()
            self.df.reset_index(drop=True, inplace=True)
        except pd.errors.ParserError as exc:
            raise ValueError(f"CSV is corrupted: {exc}") from exc

    def _validate(self, df: pd.DataFrame) -> None:
        missing = [c for c in self.REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Dataset is missing columns: {missing}")


# ═══════════════════════════════════════════════════════════════════════════════
# AI / RECOMMENDATION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class RecommendationEngine:
    """
    Content-Based Filtering using TF-IDF Vectorization + Cosine Similarity.

    Feature engineering:
        Combined text = genre (×3 weight) + keywords (×2) + description (×1)
    """

    def __init__(self, dataset: MovieDataset) -> None:
        self.dataset   = dataset
        self._vectorizer: Optional[TfidfVectorizer] = None
        self._tfidf_matrix = None
        self._is_fitted: bool = False
        self._build_model()

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def is_ready(self) -> bool:
        return self._is_fitted

    def rebuild(self, dataset: MovieDataset) -> None:
        """Rebuild the model with a new / refreshed dataset."""
        self.dataset = dataset
        self._build_model()

    def recommend(self, title: str, top_n: int = 5) -> List[Dict]:
        """
        Return top_n recommendations for a given movie title.

        Returns a list of dicts with keys:
            title, genre, description, similarity_pct, explanation
        """
        if not self._is_fitted:
            return []

        df = self.dataset.df
        title_lower = title.lower()
        matches = df[df["title"].str.lower() == title_lower]
        if matches.empty:
            return []

        idx = matches.index[0]
        sim_vector = cosine_similarity(
            self._tfidf_matrix[idx : idx + 1], self._tfidf_matrix
        ).flatten()

        # Exclude self, sort descending
        sim_scores = [
            (i, float(score))
            for i, score in enumerate(sim_vector)
            if i != idx
        ]
        sim_scores.sort(key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[:top_n]

        results = []
        base_genre_words = set(
            str(df.at[idx, "genre"]).lower().split()
            + str(df.at[idx, "keywords"]).lower().split()
        )

        for movie_idx, score in sim_scores:
            row = df.iloc[movie_idx]
            candidate_words = set(
                str(row.get("genre", "")).lower().split()
                + str(row.get("keywords", "")).lower().split()
            )
            shared = base_genre_words & candidate_words
            explanation = self._build_explanation(
                title, str(row["title"]), shared, score
            )
            results.append(
                {
                    "title":          str(row["title"]),
                    "genre":          str(row.get("genre", "")),
                    "description":    str(row.get("description", ""))[:200] + "…",
                    "year":           str(row.get("year", "N/A")),
                    "rating":         row.get("rating", "N/A"),
                    "similarity_pct": round(score * 100, 1),
                    "explanation":    explanation,
                }
            )
        return results

    # ── Private helpers ───────────────────────────────────────────────────────

    def _build_model(self) -> None:
        df = self.dataset.df
        if df.empty:
            self._is_fitted = False
            return

        corpus = df.apply(self._combine_features, axis=1)

        self._vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=8000,
            sublinear_tf=True,
        )
        self._tfidf_matrix = self._vectorizer.fit_transform(corpus)
        self._is_fitted = True

    @staticmethod
    def _combine_features(row: pd.Series) -> str:
        genre    = str(row.get("genre", "")).lower()
        keywords = str(row.get("keywords", "")).lower()
        desc     = str(row.get("description", "")).lower()
        # Weight genre 3×, keywords 2×, description 1×
        return f"{genre} {genre} {genre} {keywords} {keywords} {desc}"

    @staticmethod
    def _build_explanation(base_title: str, rec_title: str,
                           shared_words: set, score: float) -> str:
        if not shared_words:
            return f"Both '{base_title}' and '{rec_title}' share similar narrative structures and themes."
        words = ", ".join(sorted(shared_words)[:6])
        if score >= 0.75:
            strength = "strongly"
        elif score >= 0.50:
            strength = "significantly"
        else:
            strength = "moderately"
        return (
            f"Recommended because both movies {strength} share "
            f"{words} themes and elements."
        )


# ═══════════════════════════════════════════════════════════════════════════════
# PERSISTENCE – Favorites & History
# ═══════════════════════════════════════════════════════════════════════════════

class PersistenceManager:
    """Simple JSON-backed persistence for favorites and recommendation history."""

    def __init__(self) -> None:
        self._favorites: List[Dict] = self._load(FAVORITES_FILE)
        self._history:   List[Dict] = self._load(HISTORY_FILE)

    # Favorites ────────────────────────────────────────────────────────────────
    def add_favorite(self, movie: Dict) -> bool:
        title = movie.get("title", "")
        if any(f["title"] == title for f in self._favorites):
            return False
        self._favorites.insert(0, movie)
        self._save(FAVORITES_FILE, self._favorites)
        return True

    def remove_favorite(self, title: str) -> None:
        self._favorites = [f for f in self._favorites if f["title"] != title]
        self._save(FAVORITES_FILE, self._favorites)

    def is_favorite(self, title: str) -> bool:
        return any(f["title"] == title for f in self._favorites)

    def get_favorites(self) -> List[Dict]:
        return list(self._favorites)

    # History ──────────────────────────────────────────────────────────────────
    def add_history(self, query: str, recommendations: List[Dict]) -> None:
        entry = {
            "query":           query,
            "recommendations": recommendations,
            "timestamp":       datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        }
        self._history.insert(0, entry)
        self._history = self._history[:50]  # keep last 50
        self._save(HISTORY_FILE, self._history)

    def get_history(self) -> List[Dict]:
        return list(self._history)

    def clear_history(self) -> None:
        self._history = []
        self._save(HISTORY_FILE, self._history)

    # Helpers ──────────────────────────────────────────────────────────────────
    @staticmethod
    def _load(path: str) -> List:
        if not os.path.exists(path):
            return []
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return []

    @staticmethod
    def _save(path: str, data: List) -> None:
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except IOError:
            pass


# ═══════════════════════════════════════════════════════════════════════════════
# UI COMPONENT HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def make_card(parent, **kwargs) -> ctk.CTkFrame:
    """Create a styled card frame."""
    defaults = dict(
        fg_color=COLORS["card"],
        corner_radius=14,
        border_width=1,
        border_color=COLORS["border"],
    )
    defaults.update(kwargs)
    return ctk.CTkFrame(parent, **defaults)


def make_label(parent, text: str, size: int = 13, weight: str = "normal",
               color: str = COLORS["text"], **kwargs) -> ctk.CTkLabel:
    return ctk.CTkLabel(
        parent, text=text,
        font=ctk.CTkFont(family="Segoe UI", size=size, weight=weight),
        text_color=color, **kwargs
    )


def make_button(parent, text: str, command=None, width: int = 140,
                height: int = 38, color: str = COLORS["accent"],
                hover: str = COLORS["accent2"], **kwargs) -> ctk.CTkButton:
    return ctk.CTkButton(
        parent, text=text, command=command,
        width=width, height=height,
        fg_color=color, hover_color=hover,
        corner_radius=10,
        font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
        **kwargs
    )


def genre_tag_color(genre: str) -> str:
    return get_genre_color(genre)


# ═══════════════════════════════════════════════════════════════════════════════
# SPLASH SCREEN
# ═══════════════════════════════════════════════════════════════════════════════

class SplashScreen(ctk.CTkToplevel):
    """Animated splash / loading screen."""

    def __init__(self) -> None:
        super().__init__()
        self.title("")
        self.geometry("520x320")
        self.resizable(False, False)
        self.overrideredirect(True)
        self._center()
        self.configure(fg_color=COLORS["bg"])
        self._build_ui()
        self._progress = 0
        self._animate()

    def _center(self) -> None:
        self.update_idletasks()
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        x  = (sw - 520) // 2
        y  = (sh - 320) // 2
        self.geometry(f"520x320+{x}+{y}")

    def _build_ui(self) -> None:
        # Background gradient effect using frames
        grad = ctk.CTkFrame(self, fg_color=COLORS["accent"], corner_radius=0, height=4)
        grad.place(x=0, y=0, relwidth=1)

        ctk.CTkLabel(
            self, text="🎬",
            font=ctk.CTkFont(size=64),
            text_color=COLORS["accent"],
        ).place(relx=0.5, rely=0.25, anchor="center")

        ctk.CTkLabel(
            self, text="CineMind AI",
            font=ctk.CTkFont(family="Segoe UI", size=32, weight="bold"),
            text_color=COLORS["text"],
        ).place(relx=0.5, rely=0.48, anchor="center")

        ctk.CTkLabel(
            self, text="Initializing AI Recommendation Engine…",
            font=ctk.CTkFont(family="Segoe UI", size=13),
            text_color=COLORS["text_sec"],
        ).place(relx=0.5, rely=0.60, anchor="center")

        self._bar = ctk.CTkProgressBar(
            self, width=360, height=6,
            fg_color=COLORS["card"],
            progress_color=COLORS["accent"],
            corner_radius=3,
        )
        self._bar.set(0)
        self._bar.place(relx=0.5, rely=0.75, anchor="center")

        self._status = ctk.CTkLabel(
            self, text="Loading dataset…",
            font=ctk.CTkFont(family="Segoe UI", size=11),
            text_color=COLORS["text_muted"],
        )
        self._status.place(relx=0.5, rely=0.85, anchor="center")

        ctk.CTkLabel(
            self, text="Sanjeevikumar D  ·  CodSoft AI Internship",
            font=ctk.CTkFont(family="Segoe UI", size=10),
            text_color=COLORS["text_muted"],
        ).place(relx=0.5, rely=0.95, anchor="center")

    STEPS = [
        (0.15, "Loading dataset…"),
        (0.35, "Building TF-IDF vectors…"),
        (0.60, "Computing cosine similarities…"),
        (0.80, "Optimising recommendation model…"),
        (1.00, "Ready!"),
    ]

    def _animate(self) -> None:
        if self._progress < len(self.STEPS):
            val, msg = self.STEPS[self._progress]
            self._bar.set(val)
            self._status.configure(text=msg)
            self._progress += 1
            delay = 350 if self._progress < len(self.STEPS) else 500
            self.after(delay, self._animate)
        else:
            self.after(400, self.destroy)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════

class CineMindApp(ctk.CTk):
    """
    Main application window for CineMind AI.

    Sidebar navigation drives four panels:
        Home · Recommend · Favorites · History · About
    """

    def __init__(self, dataset: MovieDataset, engine: RecommendationEngine,
                 persistence: PersistenceManager) -> None:
        super().__init__()
        self.dataset     = dataset
        self.engine      = engine
        self.persistence = persistence

        # State
        self._current_movie: Optional[Dict]     = None
        self._current_panel: str                = "home"
        self._dark_mode: bool                   = True
        self._search_after_id: Optional[str]    = None
        self._dropdown_visible: bool            = False

        self._setup_window()
        self._build_layout()
        self._show_panel("home")

    # ── Window setup ──────────────────────────────────────────────────────────

    def _setup_window(self) -> None:
        self.title(APP_TITLE)
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # Center window
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        x  = max((sw - WINDOW_WIDTH)  // 2, 0)
        y  = max((sh - WINDOW_HEIGHT) // 2, 0)
        self.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}+{x}+{y}")
        self.minsize(1100, 700)
        self.configure(fg_color=COLORS["bg"])

    # ── Layout skeleton ───────────────────────────────────────────────────────

    def _build_layout(self) -> None:
        # Two-column: sidebar | main
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self._sidebar  = self._build_sidebar()
        self._main_area = ctk.CTkFrame(self, fg_color=COLORS["bg"], corner_radius=0)
        self._main_area.grid(row=0, column=1, sticky="nsew")
        self._main_area.grid_rowconfigure(0, weight=1)
        self._main_area.grid_columnconfigure(0, weight=1)

        # Build all panels (hidden by default)
        self._panel_home     = self._build_home_panel()
        self._panel_recommend = self._build_recommend_panel()
        self._panel_favorites = self._build_favorites_panel()
        self._panel_history  = self._build_history_panel()
        self._panel_about    = self._build_about_panel()

    # ── Sidebar ───────────────────────────────────────────────────────────────

    def _build_sidebar(self) -> ctk.CTkFrame:
        sidebar = ctk.CTkFrame(
            self, width=220, fg_color=COLORS["sidebar"],
            corner_radius=0, border_width=0,
        )
        sidebar.grid(row=0, column=0, sticky="nsew")
        sidebar.grid_propagate(False)

        # Logo
        logo_frame = ctk.CTkFrame(sidebar, fg_color="transparent")
        logo_frame.pack(pady=(28, 8), padx=18, fill="x")
        ctk.CTkLabel(
            logo_frame, text="🎬",
            font=ctk.CTkFont(size=30),
            text_color=COLORS["accent"],
        ).pack(side="left")
        ctk.CTkLabel(
            logo_frame, text=" CineMind",
            font=ctk.CTkFont(family="Segoe UI", size=20, weight="bold"),
            text_color=COLORS["text"],
        ).pack(side="left")

        ctk.CTkLabel(
            sidebar, text="AI Recommendation Engine",
            font=ctk.CTkFont(family="Segoe UI", size=10),
            text_color=COLORS["text_muted"],
        ).pack(pady=(0, 20))

        # Divider
        ctk.CTkFrame(sidebar, height=1, fg_color=COLORS["border"]).pack(
            fill="x", padx=14, pady=(0, 16)
        )

        # Navigation items
        nav_items = [
            ("🏠", "Home",       "home"),
            ("🎬", "Recommend",  "recommend"),
            ("❤️", "Favorites",  "favorites"),
            ("📜", "History",    "history"),
            ("ℹ️", "About",      "about"),
        ]
        self._nav_buttons: Dict[str, ctk.CTkButton] = {}
        for icon, label, key in nav_items:
            btn = self._make_nav_button(sidebar, f"  {icon}  {label}", key)
            self._nav_buttons[key] = btn

        # Bottom section
        sidebar.pack_propagate(False)
        bottom = ctk.CTkFrame(sidebar, fg_color="transparent")
        bottom.pack(side="bottom", fill="x", padx=14, pady=20)

        # Theme toggle
        ctk.CTkLabel(
            bottom, text="Theme",
            font=ctk.CTkFont(family="Segoe UI", size=11),
            text_color=COLORS["text_muted"],
        ).pack(anchor="w")

        theme_row = ctk.CTkFrame(bottom, fg_color="transparent")
        theme_row.pack(fill="x", pady=(4, 12))
        self._theme_switch = ctk.CTkSwitch(
            theme_row, text="Dark Mode",
            font=ctk.CTkFont(family="Segoe UI", size=12),
            text_color=COLORS["text_sec"],
            progress_color=COLORS["accent"],
            command=self._toggle_theme,
        )
        self._theme_switch.pack(side="left")
        self._theme_switch.select()  # start dark

        # Stats mini-card
        stats_card = ctk.CTkFrame(bottom, fg_color=COLORS["card"],
                                   corner_radius=10, border_width=1,
                                   border_color=COLORS["border"])
        stats_card.pack(fill="x")
        ctk.CTkLabel(
            stats_card,
            text=f"📊  {self.dataset.total_movies()} Movies Loaded",
            font=ctk.CTkFont(family="Segoe UI", size=11),
            text_color=COLORS["text_sec"],
        ).pack(pady=8)
        ctk.CTkLabel(
            stats_card,
            text="🧠  Model: TF-IDF + Cosine",
            font=ctk.CTkFont(family="Segoe UI", size=10),
            text_color=COLORS["text_muted"],
        ).pack(pady=(0, 8))

        return sidebar

    def _make_nav_button(self, parent, text: str, key: str) -> ctk.CTkButton:
        btn = ctk.CTkButton(
            parent, text=text, anchor="w",
            fg_color="transparent",
            hover_color=COLORS["card_hover"],
            text_color=COLORS["text_sec"],
            font=ctk.CTkFont(family="Segoe UI", size=14),
            height=42, corner_radius=10,
            command=lambda k=key: self._show_panel(k),
        )
        btn.pack(fill="x", padx=10, pady=3)
        return btn

    def _show_panel(self, key: str) -> None:
        self._current_panel = key
        panels = {
            "home":      self._panel_home,
            "recommend": self._panel_recommend,
            "favorites": self._panel_favorites,
            "history":   self._panel_history,
            "about":     self._panel_about,
        }
        for k, panel in panels.items():
            panel.grid_remove()
        panels[key].grid(row=0, column=0, sticky="nsew")

        # Highlight active button
        for k, btn in self._nav_buttons.items():
            if k == key:
                btn.configure(
                    fg_color=COLORS["accent_glow"],
                    text_color=COLORS["accent"],
                )
            else:
                btn.configure(
                    fg_color="transparent",
                    text_color=COLORS["text_sec"],
                )

        # Refresh panels that need live data
        if key == "favorites":
            self._refresh_favorites()
        if key == "history":
            self._refresh_history()

    # ─────────────────────────────────────────────────────────────────────────
    # PANEL: HOME
    # ─────────────────────────────────────────────────────────────────────────

    def _build_home_panel(self) -> ctk.CTkScrollableFrame:
        panel = ctk.CTkScrollableFrame(
            self._main_area, fg_color=COLORS["bg"], corner_radius=0,
            scrollbar_button_color=COLORS["border"],
        )
        panel.grid_columnconfigure(0, weight=1)

        # ── Hero ──────────────────────────────────────────────────────────────
        hero = ctk.CTkFrame(panel, fg_color=COLORS["card"],
                             corner_radius=20, border_width=1,
                             border_color=COLORS["border"])
        hero.grid(row=0, column=0, padx=30, pady=(30, 16), sticky="ew")
        hero.grid_columnconfigure(0, weight=1)

        # Gradient header bar
        hdr = ctk.CTkFrame(hero, fg_color=COLORS["accent"], corner_radius=0,
                            height=4)
        hdr.grid(row=0, column=0, sticky="ew", padx=0, pady=0)

        ctk.CTkLabel(
            hero, text="🎬  CineMind AI",
            font=ctk.CTkFont(family="Segoe UI", size=42, weight="bold"),
            text_color=COLORS["text"],
        ).grid(row=1, column=0, padx=30, pady=(28, 6))

        ctk.CTkLabel(
            hero,
            text="AI-Powered Personalized Movie Recommendation Engine",
            font=ctk.CTkFont(family="Segoe UI", size=16),
            text_color=COLORS["text_sec"],
        ).grid(row=2, column=0, padx=30, pady=(0, 8))

        ctk.CTkLabel(
            hero,
            text=(
                "Powered by TF-IDF Vectorization · Cosine Similarity · "
                "Content-Based Filtering"
            ),
            font=ctk.CTkFont(family="Segoe UI", size=12),
            text_color=COLORS["accent"],
        ).grid(row=3, column=0, padx=30, pady=(0, 26))

        # CTA
        cta_row = ctk.CTkFrame(hero, fg_color="transparent")
        cta_row.grid(row=4, column=0, padx=30, pady=(0, 28))
        make_button(
            cta_row, "🎬  Get Recommendations", width=200,
            command=lambda: self._show_panel("recommend"),
        ).pack(side="left", padx=8)
        make_button(
            cta_row, "🎲  Surprise Me!", width=160,
            color="#2D2B55", hover="#3D3B75",
            command=self._surprise_me,
        ).pack(side="left", padx=8)

        # ── Stats row ─────────────────────────────────────────────────────────
        stats_frame = ctk.CTkFrame(panel, fg_color="transparent")
        stats_frame.grid(row=1, column=0, padx=30, pady=(0, 16), sticky="ew")
        for i in range(4):
            stats_frame.grid_columnconfigure(i, weight=1)

        stat_data = [
            ("📊", str(self.dataset.total_movies()), "Total Movies",  COLORS["accent"]),
            ("🧠", "TF-IDF",     "Recommendation Model",              COLORS["success"]),
            ("⚡", "Active",     "AI Status",                         COLORS["warning"]),
            ("🎯", "Optimised",  "Accuracy",                          "#FF6584"),
        ]
        for col, (icon, value, label, color) in enumerate(stat_data):
            self._make_stat_card(stats_frame, icon, value, label, color, col)

        # ── How it works ──────────────────────────────────────────────────────
        how = make_card(panel)
        how.grid(row=2, column=0, padx=30, pady=(0, 16), sticky="ew")
        how.grid_columnconfigure(0, weight=1)

        make_label(
            how, "⚙️  How CineMind AI Works",
            size=17, weight="bold",
        ).grid(row=0, column=0, padx=24, pady=(20, 14), sticky="w")

        steps = [
            ("1", "Search & Select",
             "Type a movie title in the Recommend panel. Live suggestions appear instantly."),
            ("2", "TF-IDF Vectorization",
             "Text features (genre, keywords, description) are transformed into numerical vectors."),
            ("3", "Cosine Similarity",
             "The engine computes angular similarity between all movie vectors to rank matches."),
            ("4", "Ranked Results",
             "Top 5 recommendations are displayed with similarity scores and AI-generated explanations."),
        ]
        steps_row = ctk.CTkFrame(how, fg_color="transparent")
        steps_row.grid(row=1, column=0, padx=24, pady=(0, 24), sticky="ew")
        for i in range(4):
            steps_row.grid_columnconfigure(i, weight=1)

        for col, (num, title, desc) in enumerate(steps):
            sc = make_card(steps_row, fg_color=COLORS["input_bg"])
            sc.grid(row=0, column=col, padx=6, sticky="nsew")
            ctk.CTkLabel(
                sc, text=num,
                font=ctk.CTkFont(family="Segoe UI", size=22, weight="bold"),
                text_color=COLORS["accent"],
            ).pack(pady=(16, 4))
            make_label(sc, title, size=13, weight="bold").pack(pady=(0, 4))
            make_label(sc, desc, size=11, color=COLORS["text_sec"],
                       wraplength=200).pack(padx=12, pady=(0, 16))

        # ── Quick tips ────────────────────────────────────────────────────────
        tips = make_card(panel)
        tips.grid(row=3, column=0, padx=30, pady=(0, 30), sticky="ew")
        tips.grid_columnconfigure(0, weight=1)
        make_label(tips, "💡  Quick Tips", size=15, weight="bold").grid(
            row=0, column=0, padx=24, pady=(16, 8), sticky="w"
        )
        tip_texts = [
            "🎲  Use the 'Surprise Me!' button for a random movie recommendation",
            "❤️  Save your favourite movies in the Favorites panel",
            "📜  All past recommendations are stored in the History panel",
            "📤  Export any recommendation list to a TXT file",
            "🔄  Use 'Refresh Dataset' to reload if you update movies.csv",
        ]
        for tip in tip_texts:
            make_label(tips, tip, size=12, color=COLORS["text_sec"]).grid(
                row=tip_texts.index(tip) + 1, column=0,
                padx=24, pady=3, sticky="w"
            )
        ctk.CTkFrame(tips, height=1, fg_color="transparent").grid(
            row=len(tip_texts) + 1, column=0, pady=8
        )

        return panel

    def _make_stat_card(self, parent, icon: str, value: str, label: str,
                        color: str, col: int) -> None:
        card = make_card(parent)
        card.grid(row=0, column=col, padx=6, sticky="nsew")
        ctk.CTkLabel(
            card, text=icon,
            font=ctk.CTkFont(size=26),
        ).pack(pady=(18, 4))
        make_label(card, value, size=22, weight="bold", color=color).pack()
        make_label(card, label, size=11, color=COLORS["text_sec"]).pack(
            pady=(2, 18)
        )

    # ─────────────────────────────────────────────────────────────────────────
    # PANEL: RECOMMEND
    # ─────────────────────────────────────────────────────────────────────────

    def _build_recommend_panel(self) -> ctk.CTkFrame:
        panel = ctk.CTkFrame(self._main_area, fg_color=COLORS["bg"],
                              corner_radius=0)
        panel.grid_columnconfigure(0, weight=1)
        panel.grid_rowconfigure(2, weight=1)

        # ── Top bar ───────────────────────────────────────────────────────────
        topbar = ctk.CTkFrame(panel, fg_color=COLORS["sidebar"],
                               corner_radius=0, height=64)
        topbar.grid(row=0, column=0, sticky="ew")
        topbar.grid_propagate(False)
        topbar.grid_columnconfigure(1, weight=1)

        make_label(topbar, "🎬  Recommend", size=20, weight="bold").grid(
            row=0, column=0, padx=28, pady=16
        )

        btn_row = ctk.CTkFrame(topbar, fg_color="transparent")
        btn_row.grid(row=0, column=2, padx=20, pady=12)
        make_button(btn_row, "🎲 Surprise Me!", width=150,
                    color="#2D2B55", hover="#3D3B75",
                    command=self._surprise_me).pack(side="left", padx=4)
        make_button(btn_row, "🔄 Refresh", width=110,
                    color=COLORS["card"], hover=COLORS["card_hover"],
                    command=self._refresh_dataset).pack(side="left", padx=4)
        make_button(btn_row, "📤 Export", width=110,
                    color=COLORS["card"], hover=COLORS["card_hover"],
                    command=self._export_results).pack(side="left", padx=4)
        make_button(btn_row, "🗑️ Clear", width=100,
                    color=COLORS["card"], hover=COLORS["card_hover"],
                    command=self._clear_results).pack(side="left", padx=4)

        # ── Search + movie info area ──────────────────────────────────────────
        content = ctk.CTkFrame(panel, fg_color="transparent")
        content.grid(row=1, column=0, padx=24, pady=12, sticky="ew")
        content.grid_columnconfigure(0, weight=1)
        content.grid_columnconfigure(1, weight=0)

        # Search wrapper (with dropdown)
        search_wrapper = ctk.CTkFrame(content, fg_color="transparent")
        search_wrapper.grid(row=0, column=0, sticky="ew", padx=(0, 12))
        search_wrapper.grid_columnconfigure(0, weight=1)

        search_card = make_card(search_wrapper)
        search_card.grid(row=0, column=0, sticky="ew")
        search_card.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            search_card, text="🔍",
            font=ctk.CTkFont(size=18),
            text_color=COLORS["text_sec"],
        ).grid(row=0, column=0, padx=(14, 6), pady=14)

        self._search_var = tk.StringVar()
        self._search_var.trace_add("write", self._on_search_type)
        self._search_entry = ctk.CTkEntry(
            search_card,
            textvariable=self._search_var,
            placeholder_text="Search movies… e.g. Interstellar",
            fg_color="transparent",
            border_width=0,
            font=ctk.CTkFont(family="Segoe UI", size=15),
            text_color=COLORS["text"],
        )
        self._search_entry.grid(row=0, column=1, sticky="ew", pady=10)
        self._search_entry.bind("<Return>", lambda _: self._run_recommend())
        self._search_entry.bind("<Escape>", lambda _: self._hide_dropdown())

        # Dropdown listbox
        self._dropdown = ctk.CTkScrollableFrame(
            search_wrapper,
            fg_color=COLORS["card"],
            corner_radius=12,
            border_width=1,
            border_color=COLORS["border"],
            height=200,
        )
        # Not gridded until needed

        # Recommend button
        self._rec_btn = make_button(
            content, "🧠 Recommend", width=160, height=52,
            command=self._run_recommend,
        )
        self._rec_btn.grid(row=0, column=1)

        # ── Movie info card ───────────────────────────────────────────────────
        self._movie_info_frame = ctk.CTkFrame(panel, fg_color="transparent")
        self._movie_info_frame.grid(row=2, column=0, padx=24, pady=(0, 4),
                                     sticky="nsew")
        self._movie_info_frame.grid_columnconfigure(0, weight=1)
        self._movie_info_frame.grid_rowconfigure(1, weight=1)

        self._movie_detail_card  = self._build_empty_movie_card(self._movie_info_frame)
        self._results_scroll     = self._build_results_area(self._movie_info_frame)

        return panel

    def _build_empty_movie_card(self, parent) -> ctk.CTkFrame:
        card = make_card(parent)
        card.grid(row=0, column=0, pady=(0, 12), sticky="ew")
        card.grid_columnconfigure(0, weight=1)

        placeholder = make_label(
            card,
            "Search for a movie above to see its details here",
            size=13, color=COLORS["text_muted"],
        )
        placeholder.grid(padx=20, pady=20)
        self._movie_card_placeholder = placeholder
        self._movie_card_content_frame: Optional[ctk.CTkFrame] = None
        return card

    def _build_results_area(self, parent) -> ctk.CTkScrollableFrame:
        scroll = ctk.CTkScrollableFrame(
            parent, fg_color="transparent",
            scrollbar_button_color=COLORS["border"],
        )
        scroll.grid(row=1, column=0, sticky="nsew")
        scroll.grid_columnconfigure(0, weight=1)
        self._results_container = scroll
        return scroll

    # ── Search logic ──────────────────────────────────────────────────────────

    def _on_search_type(self, *_) -> None:
        if self._search_after_id:
            self.after_cancel(self._search_after_id)
        self._search_after_id = self.after(200, self._update_dropdown)

    def _update_dropdown(self) -> None:
        query = self._search_var.get()
        matches = self.dataset.search_titles(query)[:12]
        self._show_dropdown(matches)

    def _show_dropdown(self, titles: List[str]) -> None:
        # Clear old
        for w in self._dropdown.winfo_children():
            w.destroy()
        if not titles:
            self._hide_dropdown()
            return

        for title in titles:
            btn = ctk.CTkButton(
                self._dropdown,
                text=title,
                anchor="w",
                fg_color="transparent",
                hover_color=COLORS["card_hover"],
                text_color=COLORS["text"],
                font=ctk.CTkFont(family="Segoe UI", size=13),
                height=34, corner_radius=8,
                command=lambda t=title: self._select_movie_from_dropdown(t),
            )
            btn.pack(fill="x", padx=4, pady=2)

        self._dropdown.grid(row=1, column=0, sticky="ew", pady=(4, 0))
        self._dropdown_visible = True

    def _hide_dropdown(self) -> None:
        self._dropdown.grid_remove()
        self._dropdown_visible = False

    def _select_movie_from_dropdown(self, title: str) -> None:
        self._search_var.set(title)
        self._hide_dropdown()
        self._load_movie_detail(title)

    def _load_movie_detail(self, title: str) -> None:
        movie = self.dataset.get_movie_by_title(title)
        if not movie:
            return
        self._current_movie = movie
        self._render_movie_card(movie)

    # ── Movie detail card ─────────────────────────────────────────────────────

    def _render_movie_card(self, movie: Dict) -> None:
        # Clear placeholder
        self._movie_card_placeholder.grid_remove()
        if self._movie_card_content_frame:
            self._movie_card_content_frame.destroy()

        cf = ctk.CTkFrame(self._movie_detail_card, fg_color="transparent")
        cf.grid(row=1, column=0, sticky="ew", padx=0, pady=0)
        cf.grid_columnconfigure(1, weight=1)
        self._movie_card_content_frame = cf

        # Accent bar
        genre_color = genre_tag_color(str(movie.get("genre", "")))
        bar = ctk.CTkFrame(cf, fg_color=genre_color, corner_radius=0,
                            width=6)
        bar.grid(row=0, column=0, rowspan=5, sticky="ns", padx=(0, 16),
                 pady=16)

        # Title row
        title_row = ctk.CTkFrame(cf, fg_color="transparent")
        title_row.grid(row=0, column=1, sticky="ew", pady=(14, 2))
        title_row.grid_columnconfigure(0, weight=1)

        make_label(
            title_row, str(movie.get("title", "")),
            size=20, weight="bold"
        ).grid(row=0, column=0, sticky="w")

        fav_icon = "❤️" if self.persistence.is_favorite(
            str(movie.get("title", ""))
        ) else "🤍"
        self._fav_btn = make_button(
            title_row, fav_icon, width=50, height=32,
            color=COLORS["card_hover"], hover=COLORS["card"],
            command=self._toggle_favorite,
        )
        self._fav_btn.grid(row=0, column=1, padx=(8, 16))

        # Genre badge
        genre_row = ctk.CTkFrame(cf, fg_color="transparent")
        genre_row.grid(row=1, column=1, sticky="w", pady=4)
        genre_badge = ctk.CTkLabel(
            genre_row,
            text=f"  {movie.get('genre', 'Unknown')}  ",
            font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"),
            text_color=COLORS["text"],
            fg_color=genre_color,
            corner_radius=8,
        )
        genre_badge.pack(side="left")

        if "year" in movie and str(movie["year"]) != "N/A":
            make_label(
                genre_row, f"  {movie['year']}",
                size=12, color=COLORS["text_muted"],
            ).pack(side="left")

        if "rating" in movie:
            make_label(
                genre_row, f"  ⭐ {movie['rating']}",
                size=12, color=COLORS["warning"],
            ).pack(side="left")

        # Description
        desc = str(movie.get("description", ""))[:320]
        make_label(cf, desc, size=12, color=COLORS["text_sec"],
                   wraplength=760).grid(
            row=2, column=1, sticky="w", pady=(4, 8), padx=(0, 20)
        )

        # Confidence (fake AI confidence based on popularity)
        pop = int(str(movie.get("popularity", 75)))
        conf = min(100, max(60, pop))
        conf_row = ctk.CTkFrame(cf, fg_color="transparent")
        conf_row.grid(row=3, column=1, sticky="w", pady=(0, 14))

        make_label(conf_row, f"🤖 AI Confidence: ", size=12,
                   color=COLORS["text_muted"]).pack(side="left")
        ctk.CTkProgressBar(
            conf_row, width=160, height=8,
            fg_color=COLORS["input_bg"],
            progress_color=COLORS["accent"],
            corner_radius=4,
        ).pack(side="left", padx=(0, 8))
        # We set value after packing
        conf_bar = conf_row.winfo_children()[-1]  # last widget
        # Can't call set before pack – use after
        cf.after(50, lambda: conf_bar.set(conf / 100))
        make_label(conf_row, f"{conf}%", size=12, weight="bold",
                   color=COLORS["accent"]).pack(side="left")

    # ── Recommend logic ───────────────────────────────────────────────────────

    def _run_recommend(self) -> None:
        title = self._search_var.get().strip()
        if not title:
            self._show_error("Please enter or select a movie title first.")
            return
        if not self.engine.is_ready:
            self._show_error("Recommendation engine is not ready.")
            return

        movie = self.dataset.get_movie_by_title(title)
        if not movie:
            self._show_error(
                f"Movie '{title}' not found in the dataset.\n"
                "Please select a title from the search suggestions."
            )
            return

        self._load_movie_detail(title)
        self._clear_results_widgets()

        # Animate: show loading label
        loading_lbl = make_label(
            self._results_container,
            "🧠  AI is thinking…",
            size=15, color=COLORS["accent"],
        )
        loading_lbl.grid(row=0, column=0, pady=20)
        self.update()

        def _do_recommend():
            results = self.engine.recommend(title, top_n=5)
            self.after(0, lambda: self._render_results(results, title, loading_lbl))

        threading.Thread(target=_do_recommend, daemon=True).start()

    def _render_results(self, results: List[Dict], query_title: str,
                        loading_lbl) -> None:
        loading_lbl.destroy()
        if not results:
            make_label(
                self._results_container,
                "No recommendations found.",
                size=14, color=COLORS["text_muted"],
            ).grid(row=0, column=0, pady=20)
            return

        # Header
        make_label(
            self._results_container,
            f"🎯  Top {len(results)} Recommendations for  \"{query_title}\"",
            size=16, weight="bold",
        ).grid(row=0, column=0, sticky="w", pady=(4, 12))

        for i, rec in enumerate(results):
            self._make_rec_card(self._results_container, rec, i + 1)

        # Persist
        self.persistence.add_history(query_title, results)

    def _make_rec_card(self, parent, rec: Dict, rank: int) -> None:
        sim   = rec["similarity_pct"]
        color = (COLORS["success"] if sim >= 80 else
                 COLORS["warning"] if sim >= 60 else COLORS["text_sec"])
        gc    = genre_tag_color(str(rec["genre"]))

        card = make_card(parent)
        card.grid(row=rank, column=0, sticky="ew", pady=6)
        card.grid_columnconfigure(1, weight=1)

        # Rank badge
        rank_lbl = ctk.CTkLabel(
            card,
            text=f"#{rank}",
            font=ctk.CTkFont(family="Segoe UI", size=16, weight="bold"),
            text_color=COLORS["accent"],
            width=50,
        )
        rank_lbl.grid(row=0, column=0, rowspan=3, padx=(16, 0), pady=16)

        # Accent stripe
        stripe = ctk.CTkFrame(card, fg_color=gc, width=4, corner_radius=0)
        stripe.grid(row=0, column=1, rowspan=3, sticky="ns",
                    padx=(8, 12), pady=10)

        # Title + similarity badge
        row_top = ctk.CTkFrame(card, fg_color="transparent")
        row_top.grid(row=0, column=2, sticky="ew", pady=(14, 2))
        row_top.grid_columnconfigure(0, weight=1)

        make_label(
            row_top, str(rec["title"]),
            size=15, weight="bold",
        ).grid(row=0, column=0, sticky="w")

        sim_badge = ctk.CTkLabel(
            row_top,
            text=f"  {sim}% Match  ",
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
            text_color=COLORS["text"],
            fg_color=color,
            corner_radius=8,
        )
        sim_badge.grid(row=0, column=1, padx=(0, 16))

        # Genre row
        genre_row = ctk.CTkFrame(card, fg_color="transparent")
        genre_row.grid(row=1, column=2, sticky="w", pady=2)
        ctk.CTkLabel(
            genre_row,
            text=f"  {rec['genre']}  ",
            font=ctk.CTkFont(family="Segoe UI", size=11),
            text_color=COLORS["text"],
            fg_color=gc,
            corner_radius=6,
        ).pack(side="left")
        if rec.get("year"):
            make_label(genre_row, f"  {rec['year']}", size=11,
                       color=COLORS["text_muted"]).pack(side="left")
        if rec.get("rating"):
            make_label(genre_row, f"  ⭐ {rec['rating']}", size=11,
                       color=COLORS["warning"]).pack(side="left")

        # Explanation
        make_label(
            card, f"💡  {rec['explanation']}",
            size=11, color=COLORS["text_sec"],
            wraplength=700,
        ).grid(row=2, column=2, sticky="w", pady=(2, 14), padx=(0, 20))

        # Similarity bar
        bar_frame = ctk.CTkFrame(card, fg_color="transparent")
        bar_frame.grid(row=3, column=2, sticky="ew", pady=(0, 14))

        make_label(bar_frame, "Similarity:  ", size=11,
                   color=COLORS["text_muted"]).pack(side="left")
        sim_bar = ctk.CTkProgressBar(
            bar_frame, width=220, height=6,
            fg_color=COLORS["input_bg"],
            progress_color=color,
            corner_radius=3,
        )
        sim_bar.pack(side="left")
        card.after(80, lambda: sim_bar.set(sim / 100))

    # ─────────────────────────────────────────────────────────────────────────
    # PANEL: FAVORITES
    # ─────────────────────────────────────────────────────────────────────────

    def _build_favorites_panel(self) -> ctk.CTkFrame:
        panel = ctk.CTkFrame(self._main_area, fg_color=COLORS["bg"],
                              corner_radius=0)
        panel.grid_columnconfigure(0, weight=1)
        panel.grid_rowconfigure(1, weight=1)

        # Header
        hdr = ctk.CTkFrame(panel, fg_color=COLORS["sidebar"],
                             corner_radius=0, height=64)
        hdr.grid(row=0, column=0, sticky="ew")
        hdr.grid_propagate(False)
        hdr.grid_columnconfigure(1, weight=1)

        make_label(hdr, "❤️  Favourites", size=20, weight="bold").grid(
            row=0, column=0, padx=28, pady=16
        )
        make_button(
            hdr, "🗑️ Clear All", width=120,
            color=COLORS["danger"], hover="#CC0000",
            command=self._clear_all_favorites,
        ).grid(row=0, column=2, padx=20, pady=16)

        self._favorites_scroll = ctk.CTkScrollableFrame(
            panel, fg_color="transparent",
            scrollbar_button_color=COLORS["border"],
        )
        self._favorites_scroll.grid(row=1, column=0, sticky="nsew",
                                     padx=24, pady=16)
        self._favorites_scroll.grid_columnconfigure(0, weight=1)
        return panel

    def _refresh_favorites(self) -> None:
        for w in self._favorites_scroll.winfo_children():
            w.destroy()
        favs = self.persistence.get_favorites()
        if not favs:
            make_label(
                self._favorites_scroll,
                "No favourites yet.\n\nSearch for a movie and click 🤍 to add it.",
                size=14, color=COLORS["text_muted"],
            ).grid(pady=40)
            return
        for i, movie in enumerate(favs):
            self._make_fav_card(self._favorites_scroll, movie, i)

    def _make_fav_card(self, parent, movie: Dict, row: int) -> None:
        gc   = genre_tag_color(str(movie.get("genre", "")))
        card = make_card(parent)
        card.grid(row=row, column=0, sticky="ew", pady=6)
        card.grid_columnconfigure(1, weight=1)

        ctk.CTkFrame(card, fg_color=gc, width=6, corner_radius=0).grid(
            row=0, column=0, rowspan=2, sticky="ns", pady=14, padx=(0, 14)
        )
        make_label(card, str(movie.get("title", "")),
                   size=15, weight="bold").grid(
            row=0, column=1, sticky="w", pady=(14, 2)
        )
        make_label(card, str(movie.get("genre", "")),
                   size=12, color=COLORS["text_sec"]).grid(
            row=1, column=1, sticky="w", pady=(0, 14)
        )

        make_button(
            card, "✕ Remove", width=100, height=30,
            color=COLORS["danger"], hover="#CC0000",
            command=lambda t=str(movie.get("title", "")): (
                self.persistence.remove_favorite(t),
                self._refresh_favorites()
            ),
        ).grid(row=0, column=2, padx=16, pady=14)

        make_button(
            card, "🎬 Recommend", width=130, height=30,
            command=lambda t=str(movie.get("title", "")): (
                self._show_panel("recommend"),
                self._search_var.set(t),
                self._run_recommend(),
            ),
        ).grid(row=1, column=2, padx=16, pady=(0, 14))

    # ─────────────────────────────────────────────────────────────────────────
    # PANEL: HISTORY
    # ─────────────────────────────────────────────────────────────────────────

    def _build_history_panel(self) -> ctk.CTkFrame:
        panel = ctk.CTkFrame(self._main_area, fg_color=COLORS["bg"],
                              corner_radius=0)
        panel.grid_columnconfigure(0, weight=1)
        panel.grid_rowconfigure(1, weight=1)

        hdr = ctk.CTkFrame(panel, fg_color=COLORS["sidebar"],
                             corner_radius=0, height=64)
        hdr.grid(row=0, column=0, sticky="ew")
        hdr.grid_propagate(False)
        hdr.grid_columnconfigure(1, weight=1)
        make_label(hdr, "📜  History", size=20, weight="bold").grid(
            row=0, column=0, padx=28, pady=16
        )
        make_button(
            hdr, "🗑️ Clear History", width=150,
            color=COLORS["danger"], hover="#CC0000",
            command=self._clear_all_history,
        ).grid(row=0, column=2, padx=20, pady=16)

        self._history_scroll = ctk.CTkScrollableFrame(
            panel, fg_color="transparent",
            scrollbar_button_color=COLORS["border"],
        )
        self._history_scroll.grid(row=1, column=0, sticky="nsew",
                                    padx=24, pady=16)
        self._history_scroll.grid_columnconfigure(0, weight=1)
        return panel

    def _refresh_history(self) -> None:
        for w in self._history_scroll.winfo_children():
            w.destroy()
        history = self.persistence.get_history()
        if not history:
            make_label(
                self._history_scroll,
                "No history yet.\n\nGenerate your first recommendation!",
                size=14, color=COLORS["text_muted"],
            ).grid(pady=40)
            return
        for i, entry in enumerate(history):
            self._make_history_card(self._history_scroll, entry, i)

    def _make_history_card(self, parent, entry: Dict, row: int) -> None:
        card = make_card(parent)
        card.grid(row=row, column=0, sticky="ew", pady=6)
        card.grid_columnconfigure(0, weight=1)

        hdr_row = ctk.CTkFrame(card, fg_color="transparent")
        hdr_row.grid(row=0, column=0, sticky="ew", padx=16, pady=(14, 4))
        hdr_row.grid_columnconfigure(0, weight=1)

        make_label(
            hdr_row, f"🔍  {entry.get('query', 'Unknown')}",
            size=14, weight="bold",
        ).grid(row=0, column=0, sticky="w")
        make_label(
            hdr_row, str(entry.get("timestamp", "")),
            size=11, color=COLORS["text_muted"],
        ).grid(row=0, column=1)

        recs = entry.get("recommendations", [])
        titles = "  ·  ".join(r["title"] for r in recs[:5])
        make_label(card, f"🎬 {titles}", size=11,
                   color=COLORS["text_sec"]).grid(
            row=1, column=0, sticky="w", padx=16, pady=(0, 14)
        )

    # ─────────────────────────────────────────────────────────────────────────
    # PANEL: ABOUT
    # ─────────────────────────────────────────────────────────────────────────

    def _build_about_panel(self) -> ctk.CTkScrollableFrame:
        panel = ctk.CTkScrollableFrame(
            self._main_area, fg_color=COLORS["bg"], corner_radius=0,
            scrollbar_button_color=COLORS["border"],
        )
        panel.grid_columnconfigure(0, weight=1)

        # Hero
        hero = make_card(panel)
        hero.grid(row=0, column=0, padx=30, pady=(30, 20), sticky="ew")
        hero.grid_columnconfigure(0, weight=1)
        ctk.CTkFrame(hero, height=4, fg_color=COLORS["accent"],
                      corner_radius=0).grid(row=0, column=0, sticky="ew")
        ctk.CTkLabel(
            hero, text="🎬 CineMind AI",
            font=ctk.CTkFont(family="Segoe UI", size=32, weight="bold"),
            text_color=COLORS["text"],
        ).grid(row=1, column=0, pady=(20, 6))
        make_label(
            hero, "Intelligent Movie Recommendation System",
            size=16, color=COLORS["text_sec"],
        ).grid(row=2, column=0)
        make_label(
            hero, "CodSoft AI Internship – Task 4",
            size=13, color=COLORS["accent"],
        ).grid(row=3, column=0, pady=(4, 0))
        make_label(
            hero, "Developer: Sanjeevikumar D",
            size=14, weight="bold", color=COLORS["text"],
        ).grid(row=4, column=0, pady=(8, 4))
        make_label(
            hero, f"Dataset: {self.dataset.total_movies()} curated movies · v1.0.0",
            size=12, color=COLORS["text_muted"],
        ).grid(row=5, column=0, pady=(0, 20))

        # Tech stack
        tech = make_card(panel)
        tech.grid(row=1, column=0, padx=30, pady=(0, 20), sticky="ew")
        tech.grid_columnconfigure(0, weight=1)
        make_label(tech, "🛠️  Technology Stack", size=17, weight="bold").grid(
            row=0, column=0, padx=24, pady=(20, 12), sticky="w"
        )
        stack = [
            ("Python 3",      "Core programming language"),
            ("CustomTkinter", "Modern dark-themed GUI framework"),
            ("Pandas",        "Dataset loading and manipulation"),
            ("Scikit-Learn",  "TF-IDF Vectorizer & Cosine Similarity"),
            ("NumPy",         "Numerical operations"),
        ]
        for i, (name, desc) in enumerate(stack):
            row_f = ctk.CTkFrame(tech, fg_color="transparent")
            row_f.grid(row=i + 1, column=0, sticky="w", padx=24, pady=3)
            ctk.CTkLabel(
                row_f,
                text=f"  {name}  ",
                font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
                text_color=COLORS["text"],
                fg_color=COLORS["accent"],
                corner_radius=6,
            ).pack(side="left")
            make_label(row_f, f"  {desc}", size=12,
                       color=COLORS["text_sec"]).pack(side="left")
        ctk.CTkFrame(tech, height=1, fg_color="transparent").grid(
            row=len(stack) + 1, column=0, pady=8
        )

        # AI explanation
        ai_card = make_card(panel)
        ai_card.grid(row=2, column=0, padx=30, pady=(0, 20), sticky="ew")
        ai_card.grid_columnconfigure(0, weight=1)
        make_label(ai_card, "🤖  AI Methodology", size=17,
                   weight="bold").grid(
            row=0, column=0, padx=24, pady=(20, 12), sticky="w"
        )
        ai_text = (
            "CineMind AI uses Content-Based Filtering — a fundamental technique in "
            "recommender systems that analyses the intrinsic attributes of items "
            "(movies) to recommend similar ones.\n\n"
            "TF-IDF (Term Frequency-Inverse Document Frequency):\n"
            "Each movie's genre, keywords, and description are combined into a "
            "single text document. TF-IDF converts this text into a numerical "
            "vector where each dimension corresponds to a unique term weighted "
            "by its importance across the whole corpus.\n\n"
            "Cosine Similarity:\n"
            "The similarity between two movie vectors is computed as the cosine "
            "of the angle between them. A score of 1.0 means identical; 0.0 means "
            "completely unrelated. Results are ranked by descending similarity to "
            "present the top 5 most relevant recommendations."
        )
        make_label(ai_card, ai_text, size=12, color=COLORS["text_sec"],
                   wraplength=900, justify="left").grid(
            row=1, column=0, padx=24, pady=(0, 24), sticky="w"
        )

        return panel

    # ─────────────────────────────────────────────────────────────────────────
    # ACTIONS
    # ─────────────────────────────────────────────────────────────────────────

    def _toggle_favorite(self) -> None:
        if not self._current_movie:
            return
        title = str(self._current_movie.get("title", ""))
        if self.persistence.is_favorite(title):
            self.persistence.remove_favorite(title)
            self._fav_btn.configure(text="🤍")
        else:
            self.persistence.add_favorite(self._current_movie)
            self._fav_btn.configure(text="❤️")

    def _surprise_me(self) -> None:
        movie = self.dataset.get_random_movie()
        if not movie:
            return
        self._show_panel("recommend")
        self._search_var.set(str(movie["title"]))
        self._run_recommend()

    def _clear_results(self) -> None:
        self._clear_results_widgets()
        self._movie_card_placeholder.grid()
        if self._movie_card_content_frame:
            self._movie_card_content_frame.destroy()
            self._movie_card_content_frame = None
        self._search_var.set("")
        self._current_movie = None

    def _clear_results_widgets(self) -> None:
        for w in self._results_container.winfo_children():
            w.destroy()

    def _refresh_dataset(self) -> None:
        try:
            self.dataset.reload()
            self.engine.rebuild(self.dataset)
            messagebox.showinfo(
                "Dataset Refreshed",
                f"✅ Dataset reloaded successfully.\n"
                f"📊 {self.dataset.total_movies()} movies loaded."
            )
        except Exception as exc:
            self._show_error(f"Failed to reload dataset:\n{exc}")

    def _export_results(self) -> None:
        children = list(self._results_container.winfo_children())
        if not children:
            self._show_error("No recommendations to export.")
            return
        filepath = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            title="Export Recommendations",
        )
        if not filepath:
            return
        try:
            query = self._search_var.get()
            history = self.persistence.get_history()
            latest  = next(
                (h for h in history if h.get("query") == query), None
            )
            with open(filepath, "w", encoding="utf-8") as f:
                f.write("=" * 60 + "\n")
                f.write(f"CineMind AI – Recommendation Export\n")
                f.write(
                    f"Generated: {datetime.datetime.now():%Y-%m-%d %H:%M}\n"
                )
                f.write("=" * 60 + "\n\n")
                if latest:
                    f.write(f"Based on: {latest['query']}\n\n")
                    for i, rec in enumerate(latest["recommendations"], 1):
                        f.write(f"{i}. {rec['title']}\n")
                        f.write(f"   Genre:      {rec['genre']}\n")
                        f.write(f"   Similarity: {rec['similarity_pct']}%\n")
                        f.write(f"   {rec['explanation']}\n\n")
            messagebox.showinfo("Exported", f"✅ Saved to:\n{filepath}")
        except IOError as exc:
            self._show_error(f"Export failed:\n{exc}")

    def _clear_all_favorites(self) -> None:
        if messagebox.askyesno("Clear Favourites",
                                "Remove all saved favourites?"):
            for movie in self.persistence.get_favorites():
                self.persistence.remove_favorite(str(movie.get("title", "")))
            self._refresh_favorites()

    def _clear_all_history(self) -> None:
        if messagebox.askyesno("Clear History",
                                "Delete all recommendation history?"):
            self.persistence.clear_history()
            self._refresh_history()

    def _toggle_theme(self) -> None:
        self._dark_mode = not self._dark_mode
        mode = "dark" if self._dark_mode else "light"
        ctk.set_appearance_mode(mode)

    @staticmethod
    def _show_error(msg: str) -> None:
        messagebox.showerror("CineMind AI – Error", msg)


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    # Splash
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")

    root_temp = ctk.CTk()
    root_temp.withdraw()

    splash = SplashScreen()
    splash.lift()
    splash.focus_force()

    # Load data while splash runs
    error_msg: Optional[str] = None
    dataset: Optional[MovieDataset]   = None
    engine:  Optional[RecommendationEngine] = None

    def _init():
        nonlocal dataset, engine, error_msg
        try:
            dataset = MovieDataset(DATASET_PATH)
            engine  = RecommendationEngine(dataset)
        except Exception as exc:
            error_msg = str(exc)

    init_thread = threading.Thread(target=_init, daemon=True)
    init_thread.start()
    # Wait for splash to close (it auto-destroys after ~2.5s)
    root_temp.wait_window(splash)
    init_thread.join(timeout=10)

    root_temp.destroy()

    if error_msg or dataset is None or engine is None:
        # Fallback: try to create a minimal emergency dataset
        fallback_csv = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "movies.csv"
        )
        if not os.path.exists(fallback_csv):
            messagebox.showerror(
                "CineMind AI",
                f"Could not load dataset: {error_msg}\n\n"
                "Please ensure movies.csv exists in the application directory."
            )
            return
        try:
            dataset = MovieDataset(fallback_csv)
            engine  = RecommendationEngine(dataset)
        except Exception as exc:
            messagebox.showerror("CineMind AI", f"Fatal error: {exc}")
            return

    persistence = PersistenceManager()
    app = CineMindApp(dataset, engine, persistence)
    app.mainloop()


if __name__ == "__main__":
    main()
