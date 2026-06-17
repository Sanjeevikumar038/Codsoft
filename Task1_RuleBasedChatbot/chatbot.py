import sys
import os
import re
import json
import uuid
import random
import datetime
import threading
import time
from typing import List, Dict, Any, Optional

import customtkinter as ctk
from tkinter import filedialog, messagebox

# Fallback imports for voice functionality to guarantee startup on any machine
try:
    import pyttsx3
except ImportError:
    pyttsx3 = None

try:
    import speech_recognition as sr
except ImportError:
    sr = None


# Set customtkinter scaling and theme defaults
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


class NLPManager:
    """Manages rule-based Natural Language Processing, regex, and intent matching."""

    def __init__(self, intents_path: str):
        self.intents_path = intents_path
        self.intents: List[Dict[str, Any]] = []
        self.load_intents()
        self.context: Dict[str, Any] = {}

    def load_intents(self):
        """Loads intents from JSON database."""
        if os.path.exists(self.intents_path):
            try:
                with open(self.intents_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.intents = data.get("intents", [])
            except Exception as e:
                print(f"Error loading intents file: {e}")
                self.intents = []
        else:
            print(f"Intents file not found at: {self.intents_path}")
            self.intents = []

    def clean_text(self, text: str) -> str:
        """Cleans input text by lowercasing and removing special characters except spaces/plus/marks."""
        text = text.lower().strip()
        # Keep letters, numbers, and spaces. Remove punctuation except basic characters needed by patterns
        text = re.sub(r"[^\w\s\+\-]", "", text)
        return text

    def match_pattern(self, pattern: str, text: str) -> bool:
        """Determines if the pattern matches the text with boundary-aware logic."""
        pat = pattern.lower()
        txt = text.lower()
        
        # Safe escape for regex matching
        escaped = re.escape(pat)
        
        # Pattern boundary regex (checks that it's not bordered by letters/numbers)
        # e.g., "yo" in "new york" shouldn't match, but "yo" in "yo friend" should
        boundary_regex = rf"(?:^|[^a-zA-Z0-9]){escaped}(?:$|[^a-zA-Z0-9])"
        
        try:
            if re.search(boundary_regex, txt):
                return True
        except Exception:
            pass

        # Fallback word-by-word intersection check
        words = txt.split()
        if pat in words:
            return True
            
        return False

    def get_response(self, user_query: str) -> Dict[str, Any]:
        """Matches user query against predefined rules using regex and keyword scoring."""
        cleaned = self.clean_text(user_query)
        if not cleaned:
            return {
                "text": "It looks like you sent an empty message. Ask me anything about programming, AI, or internships!",
                "intent": "empty_input",
                "category": "System"
            }

        matched_intent = None
        highest_score = 0.0

        # 1. Regex & Word Pattern Matching
        for intent in self.intents:
            category = intent.get("category", "General")
            tag = intent.get("tag", "unknown")
            patterns = intent.get("patterns", [])

            for pattern in patterns:
                if self.match_pattern(pattern, cleaned):
                    # Higher scores for longer matching patterns
                    score = len(pattern) / (len(cleaned) + 1.0) + 0.5
                    if score > highest_score:
                        highest_score = score
                        matched_intent = intent

        # 2. Token overlap fallback (Keyword Detection)
        if not matched_intent:
            query_tokens = set(cleaned.split())
            for intent in self.intents:
                patterns = intent.get("patterns", [])
                for pattern in patterns:
                    pat_tokens = set(pattern.lower().split())
                    if not pat_tokens:
                        continue
                    overlap = query_tokens.intersection(pat_tokens)
                    if overlap:
                        # Simple Jaccard similarity score
                        score = len(overlap) / len(query_tokens.union(pat_tokens))
                        if score > highest_score and score > 0.25:
                            highest_score = score
                            matched_intent = intent

        # 3. Formulate response
        if matched_intent:
            responses = matched_intent.get("responses", [])
            response_text = random.choice(responses) if responses else "I matched this intent but have no response configured."
            intent_tag = matched_intent.get("tag", "matched")
            category = matched_intent.get("category", "General")

            # Dynamic date/time rendering if applicable
            if "{current_time}" in response_text:
                now = datetime.datetime.now().strftime("%I:%M %p, %A %B %d, %Y")
                response_text = response_text.replace("{current_time}", now)

            return {
                "text": response_text,
                "intent": intent_tag,
                "category": category
            }

        # Fallback for unknown queries
        return {
            "text": "I'm sorry, I don't have information about that yet. Try asking something related to programming, AI, internships, or technology.",
            "intent": "fallback",
            "category": "Fallback"
        }


class VoiceManager:
    """Manages Text-To-Speech (TTS) and Speech-To-Text (STT) inside background threads."""

    def __init__(self, tts_enabled_callback=None):
        self.tts_enabled = False
        self.tts_engine = None
        self.speech_recognizer = None
        self.mic = None
        self.is_listening = False
        self.tts_lock = threading.Lock()
        
        # Initialize TTS
        if pyttsx3:
            try:
                # On Windows, we configure pyttsx3 safely
                self.tts_engine = pyttsx3.init()
                self.tts_engine.setProperty("rate", 175)  # Moderate speaking rate
                voices = self.tts_engine.getProperty("voices")
                if voices:
                    # Select female or default voice if available
                    for voice in voices:
                        if "female" in voice.name.lower() or "zira" in voice.name.lower():
                            self.tts_engine.setProperty("voice", voice.id)
                            break
            except Exception as e:
                print(f"Failed to initialize pyttsx3: {e}")
                self.tts_engine = None

        # Initialize speech recognizer
        if sr:
            try:
                self.speech_recognizer = sr.Recognizer()
                # Reduce noise threshold sensitivity
                self.speech_recognizer.dynamic_energy_threshold = True
                self.speech_recognizer.energy_threshold = 4000
            except Exception as e:
                print(f"Failed to initialize SpeechRecognizer: {e}")
                self.speech_recognizer = None

    def speak(self, text: str):
        """Speaks the text in a separate thread if TTS is enabled."""
        if not self.tts_enabled or not self.tts_engine:
            return

        def _speak_task():
            # Acquire lock to prevent overlapping voice queries
            with self.tts_lock:
                try:
                    # Clean tags or special characters from text before speaking
                    speak_text = re.sub(r"<[^>]+>", "", text)
                    # Initialize pyttsx3 locally inside thread if needed (some platforms require this)
                    local_engine = pyttsx3.init()
                    local_engine.setProperty("rate", 175)
                    
                    # Apply selected voice parameters
                    if self.tts_engine:
                        local_engine.setProperty("voice", self.tts_engine.getProperty("voice"))

                    local_engine.say(speak_text)
                    local_engine.runAndWait()
                    # Clean up
                    local_engine.stop()
                except Exception as ex:
                    print(f"Speech error: {ex}")

        threading.Thread(target=_speak_task, daemon=True).start()

    def listen(self, callback_success, callback_error, status_update_callback):
        """Listens to microphone input in a background thread."""
        if not sr or not self.speech_recognizer:
            callback_error("Speech recognition libraries are not fully installed or configured (PyAudio missing).")
            return

        if self.is_listening:
            return
        
        self.is_listening = True

        def _listen_task():
            status_update_callback("Initializing microphone...")
            try:
                # Obtain mic instance
                with sr.Microphone() as source:
                    status_update_callback("Listening... Speak now.")
                    self.speech_recognizer.adjust_for_ambient_noise(source, duration=0.8)
                    audio = self.speech_recognizer.listen(source, timeout=6, phrase_time_limit=8)

                status_update_callback("Processing speech...")
                # Recognize voice (using google speech api)
                query = self.speech_recognizer.recognize_google(audio)
                self.is_listening = False
                callback_success(query)

            except sr.WaitTimeoutError:
                self.is_listening = False
                callback_error("Listening timed out. No speech detected.")
            except sr.UnknownValueError:
                self.is_listening = False
                callback_error("Could not understand your speech. Try speaking clearly.")
            except sr.RequestError as e:
                self.is_listening = False
                callback_error(f"Voice recognition service unavailable: {e}")
            except Exception as e:
                self.is_listening = False
                # Handles missing PyAudio or permission issues
                callback_error(f"Microphone access error: {str(e)}")

        threading.Thread(target=_listen_task, daemon=True).start()


class HistoryManager:
    """Manages storing, reloading, searching, favoriting, and exporting chat history."""

    def __init__(self, history_path: str):
        self.history_path = history_path
        self.history: List[Dict[str, Any]] = []
        self.load_history()

    def load_history(self):
        """Loads chat history from JSON."""
        if os.path.exists(self.history_path):
            try:
                with open(self.history_path, "r", encoding="utf-8") as f:
                    self.history = json.load(f)
            except Exception as e:
                print(f"Error loading history file: {e}")
                self.history = []
        else:
            self.history = []

    def save_history(self):
        """Saves chat history to JSON."""
        try:
            with open(self.history_path, "w", encoding="utf-8") as f:
                json.dump(self.history, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving history file: {e}")

    def add_message(self, sender: str, text: str, starred: bool = False) -> Dict[str, Any]:
        """Appends message to history and returns the message dict."""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %I:%M:%S %p")
        message_id = str(uuid.uuid4())
        msg = {
            "id": message_id,
            "sender": sender,
            "text": text,
            "timestamp": timestamp,
            "starred": starred
        }
        self.history.append(msg)
        self.save_history()
        return msg

    def toggle_favorite(self, message_id: str) -> bool:
        """Toggles starred status of message. Returns new status."""
        for msg in self.history:
            if msg["id"] == message_id:
                msg["starred"] = not msg["starred"]
                self.save_history()
                return msg["starred"]
        return False

    def clear_history(self):
        """Clears all messages in history."""
        self.history = []
        self.save_history()

    def search_history(self, keyword: str) -> List[Dict[str, Any]]:
        """Filters message logs containing specific keywords."""
        if not keyword.strip():
            return self.history
        
        kw = keyword.lower()
        return [msg for msg in self.history if kw in msg["text"].lower()]

    def get_favorites(self) -> List[Dict[str, Any]]:
        """Gets all starred messages."""
        return [msg for msg in self.history if msg.get("starred", False)]

    def export_history_to_txt(self, export_path: str) -> bool:
        """Exports full chat history to a formatted text file."""
        try:
            with open(export_path, "w", encoding="utf-8") as f:
                f.write("=========================================================\n")
                f.write("         INTELLIBOT AI - CHAT HISTORY EXPORT             \n")
                f.write(f"         Export Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=========================================================\n\n")

                for msg in self.history:
                    sender_label = "USER" if msg["sender"] == "user" else "INTELLIBOT"
                    starred_label = " [Starred]" if msg.get("starred", False) else ""
                    f.write(f"[{msg['timestamp']}]{starred_label} {sender_label}:\n")
                    f.write(f"{msg['text']}\n")
                    f.write("-" * 50 + "\n")
            return True
        except Exception as e:
            print(f"Error exporting chat history: {e}")
            return False


class LoadingSplashScreen(ctk.CTk):
    """Modern Splash Screen that simulates loading assets before starting the main application."""

    def __init__(self, on_loading_complete):
        super().__init__()
        self.on_loading_complete = on_loading_complete
        
        # Configure window size and settings
        self.title("IntelliBot Loading...")
        self.geometry("550x380")
        self.overrideredirect(True)  # Frameless splash screen
        self.resizable(False, False)
        
        # Center the window
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        x = (screen_width - 550) // 2
        y = (screen_height - 380) // 2
        self.geometry(f"+{x}+{y}")
        
        # Set dark theme background
        self.configure(fg_color="#0F1117")
        
        # Content layout container
        self.main_container = ctk.CTkFrame(self, fg_color="#0F1117", corner_radius=20, border_width=1, border_color="#2E303F")
        self.main_container.pack(fill="both", expand=True, padx=2, pady=2)
        
        # Header / Title Icon
        self.logo_label = ctk.CTkLabel(
            self.main_container,
            text="🤖",
            font=ctk.CTkFont(size=72)
        )
        self.logo_label.pack(pady=(45, 10))
        
        self.title_label = ctk.CTkLabel(
            self.main_container,
            text="INTELLIBOT AI",
            font=ctk.CTkFont(family="Outfit", size=32, weight="bold"),
            text_color="#6C63FF"
        )
        self.title_label.pack()

        self.tagline_label = ctk.CTkLabel(
            self.main_container,
            text="Smart Rule-Based Virtual Assistant",
            font=ctk.CTkFont(family="Inter", size=14),
            text_color="#A0A0A0"
        )
        self.tagline_label.pack(pady=(5, 35))

        # Progress bar
        self.progress_bar = ctk.CTkProgressBar(
            self.main_container,
            width=380,
            height=6,
            fg_color="#1A1D29",
            progress_color="#6C63FF",
            corner_radius=3
        )
        self.progress_bar.set(0.0)
        self.progress_bar.pack(pady=10)

        # Status text
        self.status_label = ctk.CTkLabel(
            self.main_container,
            text="Loading conversational intelligence...",
            font=ctk.CTkFont(family="Inter", size=11),
            text_color="#6C63FF"
        )
        self.status_label.pack()

        # Author details
        self.dev_label = ctk.CTkLabel(
            self.main_container,
            text="Developer: Sanjeevikumar D",
            font=ctk.CTkFont(family="Inter", size=10),
            text_color="#A0A0A0"
        )
        self.dev_label.pack(side="bottom", pady=20)

        # Start loading animation
        self.loading_step = 0
        self.after_id = None
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.simulate_loading()

    def simulate_loading(self):
        """Steps through a simulated progress load sequence to construct dynamic feel."""
        steps = [
            (0.15, "Initializing NLP rule configurations..."),
            (0.35, "Connecting regex matching models..."),
            (0.55, "Configuring Voice synthesis modules..."),
            (0.75, "Loading conversation logs from JSON..."),
            (0.95, "Starting IntelliBot UI Framework..."),
            (1.0, "Ready!")
        ]

        if self.loading_step < len(steps):
            prog, txt = steps[self.loading_step]
            self.progress_bar.set(prog)
            self.status_label.configure(text=txt)
            self.loading_step += 1
            
            # Duration intervals
            delay_ms = random.randint(300, 600)
            self.after_id = self.after(delay_ms, self.simulate_loading)
        else:
            # Self destroy and launch main app
            self.on_close()
            self.on_loading_complete()

    def on_close(self):
        if self.after_id:
            try:
                self.after_cancel(self.after_id)
            except Exception:
                pass
        self.destroy()


class IntelliBotApp(ctk.CTk):
    """Main IntelliBot AI Assistant UI window, applying Dark Glassmorphism styles."""

    def __init__(self):
        super().__init__()

        # Initialise managers
        base_dir = os.path.dirname(os.path.abspath(__file__))
        intents_path = os.path.join(base_dir, "intents.json")
        history_path = os.path.join(base_dir, "chat_history.json")

        self.nlp_manager = NLPManager(intents_path)
        self.voice_manager = VoiceManager()
        self.history_manager = HistoryManager(history_path)

        # Set up Window Attributes
        self.title("IntelliBot AI Assistant")
        self.geometry("1300x800")
        self.minsize(1100, 700)
        self.configure(fg_color=("#F0F2F6", "#0F1117"))

        # Center Window on screen
        self.center_window()
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        # Glassmorphic Colors
        self.colors = {
            "bg": ("#F0F2F6", "#0F1117"),
            "card": ("#FFFFFF", "#1A1D29"),
            "accent": "#6C63FF",
            "success": "#4CAF50",
            "border": ("#E0E4EC", "#2E303F"),
            "text": ("#1C1E21", "#FFFFFF"),
            "secondary_text": ("#606770", "#A0A0A0"),
            "bubble_user": "#6C63FF",
            "bubble_bot": ("#EAECEF", "#1A1D29"),
            "button_hover": "#5A52E5"
        }

        # Track active view/page
        self.active_nav = "home"
        self.view_containers: Dict[str, ctk.CTkFrame] = {}

        # Draw GUI Layout
        self.build_gui()

        # Display random welcome message
        self.init_welcome_message()

    def on_close(self):
        """Exits the application cleanly and stops pending tasks/synthesis."""
        try:
            if self.voice_manager and self.voice_manager.tts_engine:
                self.voice_manager.tts_engine.stop()
        except Exception:
            pass
        self.destroy()

    def center_window(self):
        self.update_idletasks()
        width = 1300
        height = 800
        x = (self.winfo_screenwidth() - width) // 2
        y = (self.winfo_screenheight() - height) // 2
        self.geometry(f"{width}x{height}+{x}+{y}")

    def build_gui(self):
        """Builds sidebar and view stack layout."""
        # Grid layout config
        self.grid_columnconfigure(0, weight=0)  # Sidebar
        self.grid_columnconfigure(1, weight=1)  # Main Content
        self.grid_rowconfigure(0, weight=1)

        # ------------------ SIDEBAR ------------------
        self.sidebar_frame = ctk.CTkFrame(
            self,
            width=240,
            corner_radius=0,
            fg_color=self.colors["card"],
            border_color=self.colors["border"],
            border_width=1
        )
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(6, weight=1)

        # App Identity in Sidebar
        self.sidebar_title_label = ctk.CTkLabel(
            self.sidebar_frame,
            text="🤖 IntelliBot AI",
            font=ctk.CTkFont(family="Outfit", size=22, weight="bold"),
            text_color=self.colors["accent"]
        )
        self.sidebar_title_label.grid(row=0, column=0, padx=20, pady=(30, 5), sticky="w")

        self.sidebar_subtitle_label = ctk.CTkLabel(
            self.sidebar_frame,
            text="Rule-Based Assistant",
            font=ctk.CTkFont(family="Inter", size=12),
            text_color=self.colors["secondary_text"]
        )
        self.sidebar_subtitle_label.grid(row=1, column=0, padx=20, pady=(0, 35), sticky="w")

        # Sidebar Buttons
        self.nav_buttons: Dict[str, ctk.CTkButton] = {}
        nav_items = [
            ("home", "🏠  Home"),
            ("chat", "💬  Chat Assistant"),
            ("favorites", "⭐  Starred Messages"),
            ("history", "📜  Full History"),
            ("about", "ℹ️  About System")
        ]

        for idx, (view_id, text) in enumerate(nav_items):
            btn = ctk.CTkButton(
                self.sidebar_frame,
                text=text,
                font=ctk.CTkFont(family="Inter", size=14, weight="normal"),
                fg_color="transparent",
                text_color=self.colors["text"],
                hover_color=("#EAECEF", "#252836"),
                anchor="w",
                height=45,
                corner_radius=8,
                command=lambda v=view_id: self.switch_view(v)
            )
            btn.grid(row=2 + idx, column=0, padx=15, pady=6, sticky="ew")
            self.nav_buttons[view_id] = btn

        # Theme Selector & TTS controls at sidebar bottom
        self.settings_frame = ctk.CTkFrame(self.sidebar_frame, fg_color="transparent")
        self.settings_frame.grid(row=7, column=0, padx=15, pady=25, sticky="ew")

        # Text-To-Speech Toggle Switch
        self.voice_switch = ctk.CTkSwitch(
            self.settings_frame,
            text="Voice Output (TTS)",
            font=ctk.CTkFont(family="Inter", size=12),
            progress_color=self.colors["accent"],
            command=self.toggle_tts
        )
        self.voice_switch.pack(anchor="w", pady=8)

        # Theme switcher (Dark / Light Mode)
        self.theme_switch = ctk.CTkSwitch(
            self.settings_frame,
            text="Light Theme",
            font=ctk.CTkFont(family="Inter", size=12),
            progress_color=self.colors["accent"],
            command=self.toggle_theme
        )
        self.theme_switch.pack(anchor="w", pady=8)

        # Separator line
        self.sep = ctk.CTkFrame(self.settings_frame, height=1, fg_color=self.colors["border"])
        self.sep.pack(fill="x", pady=10)

        # Clear History button
        self.clear_history_btn = ctk.CTkButton(
            self.settings_frame,
            text="🗑️ Clear Chat History",
            font=ctk.CTkFont(family="Inter", size=12),
            fg_color="transparent",
            text_color="#FF4D4D",
            hover_color=("#FCE8E6", "#2D1B1B"),
            height=32,
            corner_radius=8,
            command=self.clear_all_chats
        )
        self.clear_history_btn.pack(fill="x")

        # ------------------ CONTAINER WRAPPER ------------------
        self.content_container = ctk.CTkFrame(
            self,
            fg_color="transparent",
            corner_radius=0
        )
        self.content_container.grid(row=0, column=1, sticky="nsew", padx=25, pady=25)
        self.content_container.grid_columnconfigure(0, weight=1)
        self.content_container.grid_rowconfigure(0, weight=1)

        # Build each view frame
        self.build_home_view()
        self.build_chat_view()
        self.build_favorites_view()
        self.build_history_view()
        self.build_about_view()

        # Set default view
        self.switch_view("home")

    def switch_view(self, target_view: str):
        """Switches visibility of panels in main frame."""
        self.active_nav = target_view
        
        # Hide all frames
        for frame in self.view_containers.values():
            frame.grid_remove()
        
        # Show selected
        self.view_containers[target_view].grid(row=0, column=0, sticky="nsew")

        # Highlight sidebar selection
        for vid, btn in self.nav_buttons.items():
            if vid == target_view:
                btn.configure(
                    fg_color=self.colors["accent"],
                    text_color="white",
                    hover_color=self.colors["button_hover"]
                )
            else:
                btn.configure(
                    fg_color="transparent",
                    text_color=self.colors["text"],
                    hover_color=("#EAECEF", "#252836")
                )

        # Special actions on switching
        if target_view == "favorites":
            self.refresh_favorites_view()
        elif target_view == "history":
            self.refresh_history_view()

    # ------------------ 1. HOME VIEW ------------------
    def build_home_view(self):
        frame = ctk.CTkFrame(self.content_container, fg_color="transparent")
        self.view_containers["home"] = frame

        frame.grid_rowconfigure(0, weight=1)
        frame.grid_rowconfigure(1, weight=0)
        frame.grid_rowconfigure(2, weight=1)
        frame.grid_columnconfigure(0, weight=1)

        # Hero Banner
        hero_frame = ctk.CTkFrame(frame, fg_color="transparent")
        hero_frame.grid(row=0, column=0, sticky="s", pady=(0, 20))

        hero_icon = ctk.CTkLabel(hero_frame, text="🤖", font=ctk.CTkFont(size=96))
        hero_icon.pack()

        hero_title = ctk.CTkLabel(
            hero_frame,
            text="IntelliBot AI",
            font=ctk.CTkFont(family="Outfit", size=48, weight="bold"),
            text_color=self.colors["accent"]
        )
        hero_title.pack(pady=10)

        hero_subtitle = ctk.CTkLabel(
            hero_frame,
            text="Your Smart Rule-Based Virtual Assistant",
            font=ctk.CTkFont(family="Inter", size=18),
            text_color=self.colors["secondary_text"]
        )
        hero_subtitle.pack()

        # Statistics / Feature Grid Cards
        cards_frame = ctk.CTkFrame(frame, fg_color="transparent")
        cards_frame.grid(row=1, column=0, sticky="n", pady=20)
        
        features = [
            ("📚", "Knowledge Base", "Answers query logic on languages, algorithms, and technical concepts."),
            ("⚡", "Fast Responses", "Immediate match scoring with rule heuristics, no network lag."),
            ("🎤", "Voice Enabled", "Speak questions using mic capturing and hear voice replies."),
            ("🧠", "NLP Powered", "Identifies intent matching structures, greetings, and career goals.")
        ]

        for i, (icon, title, desc) in enumerate(features):
            card = ctk.CTkFrame(
                cards_frame,
                width=220,
                height=180,
                fg_color=self.colors["card"],
                border_color=self.colors["border"],
                border_width=1,
                corner_radius=12
            )
            card.grid(row=0, column=i, padx=15, pady=10)
            card.grid_propagate(False)

            c_icon = ctk.CTkLabel(card, text=icon, font=ctk.CTkFont(size=28))
            c_icon.pack(pady=(20, 5))

            c_title = ctk.CTkLabel(card, text=title, font=ctk.CTkFont(family="Outfit", size=15, weight="bold"), text_color=self.colors["text"])
            c_title.pack(pady=2)

            c_desc = ctk.CTkLabel(
                card,
                text=desc,
                font=ctk.CTkFont(family="Inter", size=11),
                text_color=self.colors["secondary_text"],
                wraplength=180,
                justify="center"
            )
            c_desc.pack(pady=5)

        # Bottom Action Button
        action_btn = ctk.CTkButton(
            frame,
            text="Start Chatting Now",
            font=ctk.CTkFont(family="Inter", size=15, weight="bold"),
            fg_color=self.colors["accent"],
            text_color="white",
            hover_color=self.colors["button_hover"],
            height=48,
            corner_radius=24,
            command=lambda: self.switch_view("chat")
        )
        action_btn.grid(row=2, column=0, sticky="n", pady=30)

    # ------------------ 2. CHAT VIEW ------------------
    def build_chat_view(self):
        frame = ctk.CTkFrame(self.content_container, fg_color="transparent")
        self.view_containers["chat"] = frame

        frame.grid_columnconfigure(0, weight=1)
        frame.grid_rowconfigure(0, weight=0)  # Search/Header bar
        frame.grid_rowconfigure(1, weight=1)  # Message Area
        frame.grid_rowconfigure(2, weight=0)  # Typing Indicator
        frame.grid_rowconfigure(3, weight=0)  # User Input Area

        # Header Search Frame
        chat_header = ctk.CTkFrame(frame, fg_color="transparent")
        chat_header.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        chat_header.grid_columnconfigure(0, weight=1)
        chat_header.grid_columnconfigure(1, weight=0)

        self.chat_search_entry = ctk.CTkEntry(
            chat_header,
            placeholder_text="🔍 Search current session messages...",
            font=ctk.CTkFont(family="Inter", size=13),
            fg_color=self.colors["card"],
            border_color=self.colors["border"],
            height=36,
            corner_radius=8
        )
        self.chat_search_entry.grid(row=0, column=0, sticky="ew", padx=(0, 10))
        self.chat_search_entry.bind("<KeyRelease>", self.on_chat_search_key)

        self.export_btn = ctk.CTkButton(
            chat_header,
            text="📤 Export Chat (.txt)",
            font=ctk.CTkFont(family="Inter", size=12, weight="normal"),
            fg_color=self.colors["card"],
            text_color=self.colors["text"],
            border_color=self.colors["border"],
            border_width=1,
            hover_color=("#EAECEF", "#252836"),
            height=36,
            corner_radius=8,
            command=self.export_chat_history
        )
        self.export_btn.grid(row=0, column=1, sticky="e")

        # Chat Bubble Container
        self.chat_scroll = ctk.CTkScrollableFrame(
            frame,
            fg_color=self.colors["card"],
            border_color=self.colors["border"],
            border_width=1,
            corner_radius=12
        )
        self.chat_scroll.grid(row=1, column=0, sticky="nsew")
        self.chat_scroll.grid_columnconfigure(0, weight=1)

        # Typing Indicator Panel
        self.typing_frame = ctk.CTkFrame(frame, fg_color="transparent", height=20)
        self.typing_frame.grid(row=2, column=0, sticky="w", padx=15, pady=2)
        self.typing_label = ctk.CTkLabel(
            self.typing_frame,
            text="",
            font=ctk.CTkFont(family="Inter", size=12, slant="italic"),
            text_color=self.colors["accent"]
        )
        self.typing_label.pack(anchor="w")

        # Input Frame (Bottom)
        input_bar = ctk.CTkFrame(frame, fg_color="transparent")
        input_bar.grid(row=3, column=0, sticky="ew", pady=(10, 0))
        input_bar.grid_columnconfigure(0, weight=1)
        input_bar.grid_columnconfigure(1, weight=0)
        input_bar.grid_columnconfigure(2, weight=0)

        self.chat_entry = ctk.CTkEntry(
            input_bar,
            placeholder_text="Ask IntelliBot something (e.g., 'What is AI?', 'Python details', 'CodSoft internship')...",
            font=ctk.CTkFont(family="Inter", size=14),
            fg_color=self.colors["card"],
            border_color=self.colors["border"],
            height=48,
            corner_radius=24
        )
        self.chat_entry.grid(row=0, column=0, sticky="ew", padx=(0, 10))
        self.chat_entry.bind("<Return>", lambda event: self.send_text_message())

        # Microphone Input button
        self.mic_btn = ctk.CTkButton(
            input_bar,
            text="🎤",
            font=ctk.CTkFont(size=18),
            fg_color=self.colors["card"],
            hover_color=("#EAECEF", "#252836"),
            border_color=self.colors["border"],
            border_width=1,
            width=48,
            height=48,
            corner_radius=24,
            command=self.activate_voice_input
        )
        self.mic_btn.grid(row=0, column=1, sticky="e", padx=(0, 10))

        # Send button
        self.send_btn = ctk.CTkButton(
            input_bar,
            text="Send",
            font=ctk.CTkFont(family="Inter", size=14, weight="bold"),
            fg_color=self.colors["accent"],
            text_color="white",
            hover_color=self.colors["button_hover"],
            width=90,
            height=48,
            corner_radius=24,
            command=self.send_text_message
        )
        self.send_btn.grid(row=0, column=2, sticky="e")

        # Render loaded message history in bubble container
        self.render_chat_history()

    def render_chat_history(self, filter_kw: Optional[str] = None):
        """Draws messages in scroll frame. If filter_kw is supplied, filters search."""
        # Clear existing bubbles
        for widget in self.chat_scroll.winfo_children():
            widget.destroy()

        messages = self.history_manager.history
        if filter_kw:
            messages = self.history_manager.search_history(filter_kw)

        for msg in messages:
            self.draw_bubble(msg)
        
        self.scroll_chat_to_bottom()

    def draw_bubble(self, msg: Dict[str, Any]):
        """Renders single message bubble in scroll view."""
        is_user = msg["sender"] == "user"
        text = msg["text"]
        timestamp = msg["timestamp"]
        msg_id = msg["id"]
        starred = msg.get("starred", False)

        # Bubble wrapper to handle margins
        bubble_wrapper = ctk.CTkFrame(self.chat_scroll, fg_color="transparent")
        bubble_wrapper.pack(fill="x", padx=10, pady=8)
        bubble_wrapper.grid_columnconfigure(0, weight=1)

        # Inner container bubble
        bubble = ctk.CTkFrame(
            bubble_wrapper,
            fg_color=self.colors["bubble_user"] if is_user else self.colors["bubble_bot"][1] if self.cget("fg_color") == self.colors["bg"][1] else self.colors["bubble_bot"][0],
            corner_radius=16,
            border_color=self.colors["border"] if not is_user else self.colors["accent"],
            border_width=1
        )

        # Align left or right
        if is_user:
            bubble.grid(row=0, column=0, sticky="e", padx=(80, 0))
        else:
            bubble.grid(row=0, column=0, sticky="w", padx=(0, 80))

        # Horizontal layouts for text and sub-actions
        bubble.grid_columnconfigure(0, weight=1)
        bubble.grid_rowconfigure(0, weight=1)
        bubble.grid_rowconfigure(1, weight=0)

        # Text label
        msg_label = ctk.CTkLabel(
            bubble,
            text=text,
            font=ctk.CTkFont(family="Inter", size=13),
            text_color="white" if is_user else self.colors["text"][1] if self.cget("fg_color") == self.colors["bg"][1] else self.colors["text"][0],
            wraplength=600,
            justify="left",
            anchor="w"
        )
        msg_label.grid(row=0, column=0, padx=(16, 16), pady=(12, 6), sticky="w")

        # Info row (Time + Actions)
        info_row = ctk.CTkFrame(bubble, fg_color="transparent")
        info_row.grid(row=1, column=0, padx=12, pady=(0, 8), sticky="ew")
        
        time_lbl = ctk.CTkLabel(
            info_row,
            text=timestamp,
            font=ctk.CTkFont(family="Inter", size=10),
            text_color="#B0B5C0" if is_user else self.colors["secondary_text"][1] if self.cget("fg_color") == self.colors["bg"][1] else self.colors["secondary_text"][0]
        )
        time_lbl.pack(side="left", padx=4)

        # Star toggle button
        star_text = "★ Starred" if starred else "☆ Star"
        star_color = "#E9C46A" if starred else self.colors["secondary_text"][1] if self.cget("fg_color") == self.colors["bg"][1] else self.colors["secondary_text"][0]
        
        star_btn = ctk.CTkButton(
            info_row,
            text=star_text,
            font=ctk.CTkFont(family="Inter", size=10, weight="normal"),
            text_color=star_color,
            fg_color="transparent",
            hover_color=("#D0D5DD", "#2A2D3D") if not is_user else "#5A52E5",
            width=50,
            height=20,
            corner_radius=4,
            command=lambda mid=msg_id: self.toggle_star(mid)
        )
        star_btn.pack(side="right", padx=4)

        # For bot responses, show a "Speak" button to read it back manually
        if not is_user:
            speak_btn = ctk.CTkButton(
                info_row,
                text="🔊 Speak",
                font=ctk.CTkFont(family="Inter", size=10, weight="normal"),
                text_color=self.colors["accent"],
                fg_color="transparent",
                hover_color=("#D0D5DD", "#2A2D3D"),
                width=50,
                height=20,
                corner_radius=4,
                command=lambda t=text: self.voice_manager.speak(t)
            )
            speak_btn.pack(side="right", padx=4)

    def scroll_chat_to_bottom(self):
        """Auto-scroll the chat view down."""
        self.update_idletasks()
        # Scroll logic inside customtkinter scrollable frame
        self.chat_scroll._parent_canvas.yview_moveto(1.0)

    # ------------------ 3. FAVORITES VIEW ------------------
    def build_favorites_view(self):
        frame = ctk.CTkFrame(self.content_container, fg_color="transparent")
        self.view_containers["favorites"] = frame

        frame.grid_columnconfigure(0, weight=1)
        frame.grid_rowconfigure(0, weight=0)  # Title
        frame.grid_rowconfigure(1, weight=1)  # List frame

        title_lbl = ctk.CTkLabel(
            frame,
            text="⭐ Starred Messages",
            font=ctk.CTkFont(family="Outfit", size=24, weight="bold"),
            text_color=self.colors["accent"]
        )
        title_lbl.grid(row=0, column=0, sticky="w", pady=(0, 15))

        self.favs_scroll = ctk.CTkScrollableFrame(
            frame,
            fg_color=self.colors["card"],
            border_color=self.colors["border"],
            border_width=1,
            corner_radius=12
        )
        self.favs_scroll.grid(row=1, column=0, sticky="nsew")
        self.favs_scroll.grid_columnconfigure(0, weight=1)

    def refresh_favorites_view(self):
        """Re-draws all starred messages."""
        for widget in self.favs_scroll.winfo_children():
            widget.destroy()

        fav_msgs = self.history_manager.get_favorites()
        if not fav_msgs:
            empty_lbl = ctk.CTkLabel(
                self.favs_scroll,
                text="No starred messages yet. Star some messages in the Chat Assistant view!",
                font=ctk.CTkFont(family="Inter", size=14, slant="italic"),
                text_color=self.colors["secondary_text"]
            )
            empty_lbl.pack(pady=40)
            return

        for msg in fav_msgs:
            self.draw_fav_item(msg)

    def draw_fav_item(self, msg: Dict[str, Any]):
        """Draws item in favorites screen list."""
        sender_lbl = "USER" if msg["sender"] == "user" else "INTELLIBOT"
        bg_col = "#25293A" if self.cget("fg_color") == self.colors["bg"][1] else "#F8FAFC"
        
        card = ctk.CTkFrame(
            self.favs_scroll,
            fg_color=bg_col,
            border_color=self.colors["border"],
            border_width=1,
            corner_radius=10
        )
        card.pack(fill="x", padx=10, pady=6)
        card.grid_columnconfigure(0, weight=1)
        card.grid_rowconfigure(0, weight=0)

        # Header of card (Sender + Date + Remove button)
        card_header = ctk.CTkFrame(card, fg_color="transparent")
        card_header.grid(row=0, column=0, sticky="ew", padx=15, pady=(10, 4))
        
        badge_color = self.colors["accent"] if msg["sender"] == "user" else self.colors["success"]
        badge = ctk.CTkLabel(
            card_header,
            text=f"  {sender_lbl}  ",
            font=ctk.CTkFont(family="Inter", size=10, weight="bold"),
            text_color="white",
            fg_color=badge_color,
            corner_radius=4
        )
        badge.pack(side="left")

        time_lbl = ctk.CTkLabel(
            card_header,
            text=msg["timestamp"],
            font=ctk.CTkFont(family="Inter", size=11),
            text_color=self.colors["secondary_text"]
        )
        time_lbl.pack(side="left", padx=10)

        unstar_btn = ctk.CTkButton(
            card_header,
            text="Unstar",
            font=ctk.CTkFont(family="Inter", size=10),
            text_color="#FF4D4D",
            fg_color="transparent",
            hover_color=("#FCE8E6", "#2D1B1B"),
            width=50,
            height=20,
            corner_radius=4,
            command=lambda mid=msg["id"]: self.remove_favorite(mid)
        )
        unstar_btn.pack(side="right")

        # Text of card
        txt_lbl = ctk.CTkLabel(
            card,
            text=msg["text"],
            font=ctk.CTkFont(family="Inter", size=13),
            text_color=self.colors["text"],
            wraplength=700,
            justify="left",
            anchor="w"
        )
        txt_lbl.grid(row=1, column=0, sticky="w", padx=15, pady=(2, 12))

    # ------------------ 4. HISTORY VIEW ------------------
    def build_history_view(self):
        frame = ctk.CTkFrame(self.content_container, fg_color="transparent")
        self.view_containers["history"] = frame

        frame.grid_columnconfigure(0, weight=1)
        frame.grid_rowconfigure(0, weight=0)  # Search Bar
        frame.grid_rowconfigure(1, weight=1)  # History Logs

        search_header = ctk.CTkFrame(frame, fg_color="transparent")
        search_header.grid(row=0, column=0, sticky="ew", pady=(0, 15))
        search_header.grid_columnconfigure(0, weight=1)

        self.hist_search_entry = ctk.CTkEntry(
            search_header,
            placeholder_text="🔍 Search terms across all conversations...",
            font=ctk.CTkFont(family="Inter", size=13),
            fg_color=self.colors["card"],
            border_color=self.colors["border"],
            height=38,
            corner_radius=8
        )
        self.hist_search_entry.grid(row=0, column=0, sticky="ew", padx=(0, 10))
        self.hist_search_entry.bind("<KeyRelease>", self.on_history_search_key)

        export_hist_btn = ctk.CTkButton(
            search_header,
            text="Export All Logs",
            font=ctk.CTkFont(family="Inter", size=12, weight="bold"),
            fg_color=self.colors["accent"],
            text_color="white",
            hover_color=self.colors["button_hover"],
            height=38,
            corner_radius=8,
            command=self.export_chat_history
        )
        export_hist_btn.grid(row=0, column=1, sticky="e")

        self.history_scroll = ctk.CTkScrollableFrame(
            frame,
            fg_color=self.colors["card"],
            border_color=self.colors["border"],
            border_width=1,
            corner_radius=12
        )
        self.history_scroll.grid(row=1, column=0, sticky="nsew")
        self.history_scroll.grid_columnconfigure(0, weight=1)

    def refresh_history_view(self, search_kw: Optional[str] = None):
        """Redraws full history log."""
        for widget in self.history_scroll.winfo_children():
            widget.destroy()

        if search_kw:
            messages = self.history_manager.search_history(search_kw)
        else:
            messages = self.history_manager.history

        if not messages:
            empty_lbl = ctk.CTkLabel(
                self.history_scroll,
                text="No message logs match the criteria.",
                font=ctk.CTkFont(family="Inter", size=14, slant="italic"),
                text_color=self.colors["secondary_text"]
            )
            empty_lbl.pack(pady=40)
            return

        for msg in messages:
            self.draw_history_row(msg)

    def draw_history_row(self, msg: Dict[str, Any]):
        """Draws basic history cell block."""
        is_user = msg["sender"] == "user"
        badge_lbl = "USER" if is_user else "BOT"
        bg_col = "#202434" if self.cget("fg_color") == self.colors["bg"][1] else "#F1F5F9"
        
        row_frame = ctk.CTkFrame(
            self.history_scroll,
            fg_color=bg_col,
            corner_radius=6
        )
        row_frame.pack(fill="x", padx=10, pady=4)
        row_frame.grid_columnconfigure(2, weight=1)

        time_lbl = ctk.CTkLabel(
            row_frame,
            text=msg["timestamp"],
            font=ctk.CTkFont(family="Inter", size=11),
            text_color=self.colors["secondary_text"],
            width=170
        )
        time_lbl.grid(row=0, column=0, padx=10, pady=8, sticky="w")

        badge = ctk.CTkLabel(
            row_frame,
            text=badge_lbl,
            font=ctk.CTkFont(family="Inter", size=10, weight="bold"),
            text_color="white",
            fg_color=self.colors["accent"] if is_user else self.colors["success"],
            width=50,
            corner_radius=4
        )
        badge.grid(row=0, column=1, padx=10, pady=8)

        text_lbl = ctk.CTkLabel(
            row_frame,
            text=msg["text"],
            font=ctk.CTkFont(family="Inter", size=12),
            text_color=self.colors["text"],
            wraplength=600,
            justify="left",
            anchor="w"
        )
        text_lbl.grid(row=0, column=2, padx=15, pady=8, sticky="w")

    # ------------------ 5. ABOUT VIEW ------------------
    def build_about_view(self):
        frame = ctk.CTkFrame(self.content_container, fg_color="transparent")
        self.view_containers["about"] = frame

        frame.grid_columnconfigure(0, weight=1)
        frame.grid_rowconfigure(0, weight=0)
        frame.grid_rowconfigure(1, weight=1)

        title_lbl = ctk.CTkLabel(
            frame,
            text="ℹ️ About IntelliBot System",
            font=ctk.CTkFont(family="Outfit", size=24, weight="bold"),
            text_color=self.colors["accent"]
        )
        title_lbl.grid(row=0, column=0, sticky="w", pady=(0, 20))

        content_card = ctk.CTkFrame(
            frame,
            fg_color=self.colors["card"],
            border_color=self.colors["border"],
            border_width=1,
            corner_radius=16
        )
        content_card.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        content_card.grid_columnconfigure(0, weight=1)

        # App specs
        app_title = ctk.CTkLabel(
            content_card,
            text="IntelliBot AI",
            font=ctk.CTkFont(family="Outfit", size=28, weight="bold"),
            text_color=self.colors["accent"]
        )
        app_title.pack(pady=(35, 5))

        app_subtitle = ctk.CTkLabel(
            content_card,
            text="Smart Rule-Based Virtual Assistant (v1.0.0)",
            font=ctk.CTkFont(family="Inter", size=14, weight="normal"),
            text_color=self.colors["text"]
        )
        app_subtitle.pack(pady=2)

        sep = ctk.CTkFrame(content_card, height=1, width=400, fg_color=self.colors["border"])
        sep.pack(pady=20)

        # Developer / Project Card details
        details_frame = ctk.CTkFrame(content_card, fg_color="transparent")
        details_frame.pack(pady=10)

        details = [
            ("Developer:", "Sanjeevikumar D"),
            ("Internship:", "CodSoft Artificial Intelligence Internship"),
            ("Task ID:", "Task 1 – Rule Based Chatbot"),
            ("Tech Stack:", "Python, CustomTkinter, JSON, PyTTSx3, SpeechRecognition"),
            ("Architecture:", "Pattern-Matching Intent Classifier (Zero-API Dependency)")
        ]

        for label, val in details:
            row = ctk.CTkFrame(details_frame, fg_color="transparent")
            row.pack(fill="x", pady=6)
            
            lbl_w = ctk.CTkLabel(
                row,
                text=label.ljust(15),
                font=ctk.CTkFont(family="Inter", size=13, weight="bold"),
                text_color=self.colors["accent"],
                width=120,
                anchor="w"
            )
            lbl_w.pack(side="left")

            val_w = ctk.CTkLabel(
                row,
                text=val,
                font=ctk.CTkFont(family="Inter", size=13),
                text_color=self.colors["text"],
                anchor="w"
            )
            val_w.pack(side="left", padx=10)

        sep2 = ctk.CTkFrame(content_card, height=1, width=400, fg_color=self.colors["border"])
        sep2.pack(pady=20)

        # Description text block
        desc_lbl = ctk.CTkLabel(
            content_card,
            text="IntelliBot AI is built completely without machine learning API connections. "
                 "It showcases deterministic, rules-based natural language patterns. It has a high-fidelity "
                 "user experience resembling premium LLM chat portals, integrating offline local "
                 "Text-To-Speech playback and thread-safe voice input decoding.",
            font=ctk.CTkFont(family="Inter", size=12),
            text_color=self.colors["secondary_text"],
            wraplength=600,
            justify="center"
        )
        desc_lbl.pack(pady=(5, 35))

    # ------------------ EVENT HANDLERS & ACTIONS ------------------
    def init_welcome_message(self):
        """Appends random welcoming message on application startup."""
        if not self.history_manager.history:
            welcomes = [
                "Hello! I am IntelliBot AI, your custom assistant. How can I help you today?",
                "Welcome! Ask me anything about programming, career tips, or Artificial Intelligence.",
                "Greetings! I am ready to answer questions about Python, Java, CodSoft, and more. How can I assist?",
                "Hi there! I am running on a fully offline rule-based logic framework. Let's chat!"
            ]
            welcome_text = random.choice(welcomes)
            self.history_manager.add_message("bot", welcome_text)
            self.render_chat_history()

    def send_text_message(self):
        """Sends the message written in input entry box."""
        raw_text = self.chat_entry.get().strip()
        if not raw_text:
            return

        # Clear text box
        self.chat_entry.delete(0, "end")

        # 1. User Message
        user_msg = self.history_manager.add_message("user", raw_text)
        self.draw_bubble(user_msg)
        self.scroll_chat_to_bottom()

        # 2. Trigger typing animation and fetch response
        self.typing_label.configure(text="IntelliBot is typing...")
        
        def _get_bot_reply():
            # Simulate slight human typing latency (400 - 800ms)
            time.sleep(random.randint(4, 8) / 10.0)
            
            # Fetch rule-based reply
            reply = self.nlp_manager.get_response(raw_text)
            bot_text = reply["text"]

            # Add to storage database
            bot_msg = self.history_manager.add_message("bot", bot_text)
            
            # Draw bubble in main GUI thread
            self.after(0, lambda: self.finish_bot_reply(bot_msg))

        threading.Thread(target=_get_bot_reply, daemon=True).start()

    def finish_bot_reply(self, bot_msg: Dict[str, Any]):
        """Finished typing status, draws response, and speaks it if TTS is enabled."""
        self.typing_label.configure(text="")
        self.draw_bubble(bot_msg)
        self.scroll_chat_to_bottom()

        # Voice synthesis output
        self.voice_manager.speak(bot_msg["text"])

    def activate_voice_input(self):
        """Starts capturing microphone voice entry in background thread."""
        if not sr:
            messagebox.showwarning(
                "Voice Input Unavailable",
                "SpeechRecognition libraries are missing or not fully configured. "
                "Ensure PyAudio is installed on your Windows machine to use microphone inputs."
            )
            return

        self.mic_btn.configure(fg_color=self.colors["success"], text="🔴")
        
        def _on_success(query: str):
            self.after(0, lambda: self.reset_mic_button())
            self.after(0, lambda: self.handle_spoken_query(query))

        def _on_error(error_msg: str):
            self.after(0, lambda: self.reset_mic_button())
            self.after(0, lambda: messagebox.showerror("Voice Input Error", error_msg))

        def _on_status(status_txt: str):
            self.after(0, lambda: self.typing_label.configure(text=status_txt))

        self.voice_manager.listen(_on_success, _on_error, _on_status)

    def reset_mic_button(self):
        self.mic_btn.configure(fg_color=self.colors["card"], text="🎤")
        self.typing_label.configure(text="")

    def handle_spoken_query(self, query: str):
        """Inserts recognized spoken speech query in input entry and sends it."""
        self.chat_entry.delete(0, "end")
        self.chat_entry.insert(0, query)
        self.send_text_message()

    def toggle_tts(self):
        """Toggles state of Text To Speech voice reply option."""
        self.voice_manager.tts_enabled = self.voice_switch.get() == 1

    def toggle_theme(self):
        """Toggles theme state of customtkinter app."""
        if self.theme_switch.get() == 1:
            ctk.set_appearance_mode("Light")
            self.configure(fg_color=self.colors["bg"][0])
        else:
            ctk.set_appearance_mode("Dark")
            self.configure(fg_color=self.colors["bg"][1])
        
        # Redraw components matching theme
        self.sidebar_frame.configure(fg_color=self.colors["card"])
        self.content_container.configure(fg_color="transparent")
        
        # Refresh widgets views
        self.render_chat_history()
        self.refresh_favorites_view()
        self.refresh_history_view()

    def toggle_star(self, msg_id: str):
        """Toggles star state on single chat bubble."""
        self.history_manager.toggle_favorite(msg_id)
        # Re-render messages to show new star statuses
        self.render_chat_history()

    def remove_favorite(self, msg_id: str):
        """Unstars message from Favorites screen."""
        self.history_manager.toggle_favorite(msg_id)
        self.refresh_favorites_view()
        self.render_chat_history()

    def clear_all_chats(self):
        """Clear chat history logic."""
        confirm = messagebox.askyesno(
            "Clear History",
            "Are you sure you want to permanently delete all conversation logs?"
        )
        if confirm:
            self.history_manager.clear_history()
            self.init_welcome_message()
            self.render_chat_history()
            self.refresh_favorites_view()
            self.refresh_history_view()

    def export_chat_history(self):
        """Export txt file of the conversation history."""
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")],
            title="Export Chat History"
        )
        if file_path:
            success = self.history_manager.export_history_to_txt(file_path)
            if success:
                messagebox.showinfo("Export Successful", f"Chat history successfully saved to:\n{file_path}")
            else:
                messagebox.showerror("Export Failed", "Could not export chat log. Check files permissions.")

    def on_chat_search_key(self, event):
        """Triggers search query typing filter inside current session."""
        kw = self.chat_search_entry.get().strip()
        self.render_chat_history(filter_kw=kw if kw else None)

    def on_history_search_key(self, event):
        """Triggers search query typing filter inside full logs."""
        kw = self.hist_search_entry.get().strip()
        self.refresh_history_view(search_kw=kw if kw else None)


def main():
    """Application entry point."""
    # Run application
    splash = LoadingSplashScreen(on_loading_complete=lambda: IntelliBotApp().mainloop())
    splash.mainloop()


if __name__ == "__main__":
    main()
