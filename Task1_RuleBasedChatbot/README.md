# 🤖 IntelliBot AI — Smart Rule-Based Virtual Assistant

**IntelliBot AI** is a premium, feature-rich desktop virtual assistant designed and implemented for **CodSoft Artificial Intelligence Internship (Task 1)**. It features a modern ChatGPT-style dark glassmorphism interface, offline Text-To-Speech (TTS), speech-to-text voice recognition, favorite bookmarking system, full searchable logs, and dynamic layout customization.

---

## 🌟 Features

*   **Premium Glassmorphic Design:** Elegant UI constructed using HSL-matched dark palettes (`#0F1117`, `#1A1D29`) with CustomTkinter, styled with rounded corners and glowing border definitions.
*   **Zero-API Rule-Based Intelligence:** Built from the ground up without using external large language models (like OpenAI, Gemini, or Claude). Responses are determined entirely locally using deterministic regexes and keyword overlap logic.
*   **Dual Voice Support:**
    *   **Text-to-Speech (TTS):** Speaks responses aloud in background threads using PyTTSx3 (fully toggleable).
    *   **Speech-to-Text (STT):** Dictate queries to the assistant via the voice microphone button.
*   **Splash Loading Animation:** Elegant animated splash loader on startup.
*   **Adaptive Theme Switcher:** Smooth toggling between dark glassmorphism and clear light mode configurations.
*   **Smart Chat Persistence:** Stores session logs in `chat_history.json`, reloading conversation states upon launch.
*   **Advanced Logs Control:**
    *   **Bookmarking:** Star important messages to keep track of them in the **Starred Messages** list.
    *   **Search Engine:** Live typing search across current conversation messages or historical session databases.
    *   **Export:** Download full chat logs as clean `.txt` documentation.
    *   **Clear chat:** Instantly purge conversation history safely.

---

## 📚 Technical Concepts

### 1. Rule-Based Natural Language Processing (NLP)
Rule-based NLP maps human queries to structured intents without training deep learning weight networks. When a user sends a message, it undergoes cleaning (lowercasing, trimming, and punctuation removal).

```
[User Input] -> [Text Cleaning] -> [Intent Matcher] -> [Select Response] -> [TTS Output]
                                         |
                       +-----------------+-----------------+
                       |                                   |
            [Exact Phrase Match]                 [Token Intersection]
           (Regex boundary checks)             (Jaccard overlap coefficient)
```

### 2. Regex and Keyword Pattern Matching Heuristics
The intent analyzer performs a two-pass matching sequence:
*   **Regex Phrase Matcher:** Searches patterns using boundary checks `\b(pattern)\b` to prevent false positive substrings (e.g., ensuring "hi" does not match "history" or "shift").
*   **Jaccard Similarity Coefficient (Fallback):** If no direct regex pattern is found, the system splits inputs and patterns into tokens, computing token intersection divided by token union:
    $$J(A, B) = \frac{|A \cap B|}{|A \cup B|}$$
    If this score exceeds a default threshold, the assistant classifies the input under that intent.

---

## 📁 Project Structure

```text
Task1_RuleBasedChatbot/
│
├── chatbot.py           # Core application containing UI, managers, and logic
├── intents.json         # Conversational intents, keywords, and responses database
├── chat_history.json    # JSON storage containing previous conversation messages
├── requirements.txt     # List of python package dependencies
├── README.md            # Extensive setup and project documentation
├── assets/              # App icon resources and logos
└── screenshots/         # UI preview files
```

---

## ⚙️ Installation & Setup

### Prerequisites
1.  **Python 3.8+** installed on your Windows machine.
2.  **PyAudio pre-requisites:** Speech Recognition requires PyAudio. On Windows, if installing PyAudio via pip raises compiler issues, install it using pre-compiled wheels:
    ```bash
    pip install pipwin
    pipwin install pyaudio
    ```

### Step-by-Step Installation
1.  Clone the repository or navigate to your task folder:
    ```bash
    cd Task1_RuleBasedChatbot
    ```
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

---

## 🚀 Usage

Launch the virtual assistant by running the main Python script:
```bash
python chatbot.py
```

### Navigating the App
*   **Home Screen:** View the stats cards showing features and start a chat session.
*   **Chat Assistant:** Type queries in the entry or press the 🎤 button to record. You can search current chat messages in the top search bar or export the log.
*   **Starred Messages:** Access a compiled checklist of all messages you have bookmarked.
*   **Full History:** Check all logged chats, filter records, or perform a search.
*   **About System:** Read details on developers, technologies, and internship requirements.

---

## 🖼️ Screenshots

*Add application images here inside the `/screenshots` directory.*

---

## 🔮 Future Roadmaps
*   **Custom Regex Patterns Editor:** Add a GUI editor inside the app allowing users to update and register new regex patterns to `intents.json` directly.
*   **Enhanced POS Tagging:** Introduce tokenizers like NLTK to perform basic Part-Of-Speech parsing to improve question matching.
*   **More Voice Styles:** Support downloading and selecting multiple SAPI5 voice packs.

---

## 👨‍💻 Developer Information

*   **Developer Name:** Sanjeevikumar D
*   **Internship Program:** CodSoft Artificial Intelligence Internship
*   **Task Title:** Task 1 – Rule Based Chatbot
*   **Project Title:** IntelliBot AI – Smart Rule-Based Virtual Assistant
*   **LinkedIN Profile:** [Sanjeevikumar D](https://www.linkedin.com/) *(Add your custom profile link)*
*   **GitHub Repository:** [Sanjeevikumar D GitHub](https://github.com/) *(Add repository path)*
