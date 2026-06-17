<div align="center">

<img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/CustomTkinter-5.2+-6C63FF?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Scikit--Learn-AI%20Powered-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
<img src="https://img.shields.io/badge/CodSoft-AI%20Internship-4CAF50?style=for-the-badge"/>

# 🎬 CineMind AI
### Intelligent Movie Recommendation System
**CodSoft Artificial Intelligence Internship – Task 4**

*A premium AI-powered desktop application for personalised movie recommendations*

---

</div>

## 📌 Project Overview

**CineMind AI** is a professional-grade movie recommendation system built as part of the **CodSoft AI Internship (Task 4)**. It uses **Content-Based Filtering** powered by **TF-IDF Vectorization** and **Cosine Similarity** to analyse movie attributes (genre, keywords, description) and surface highly relevant personalised recommendations.

The application is designed to look and feel like a **commercial product** — with a dark Netflix/Spotify-inspired UI, smooth animations, glassmorphism cards, sidebar navigation, and a full feature set including favourites, history, export, and theme switching.

---

## ✨ Features

| Feature | Description |
|--------|-------------|
| 🧠 **AI Recommendations** | TF-IDF + Cosine Similarity content-based engine |
| 🔍 **Live Search** | Instant search with auto-complete dropdown |
| 🎯 **Similarity Scores** | Each recommendation shown with a % match |
| 💡 **AI Explanations** | Natural language reason for every recommendation |
| ❤️ **Favourites** | Save and manage your favourite movies |
| 📜 **History** | All past recommendation sessions persisted |
| 🎲 **Surprise Me** | Get recommendations for a random movie |
| 📤 **Export** | Export recommendations to a `.txt` file |
| 🌙 **Dark / ☀️ Light Mode** | One-click theme switching |
| 🔄 **Live Dataset Reload** | Refresh movies.csv without restarting |
| 🎞️ **Splash Screen** | Animated loading screen on startup |
| 🏛️ **Sidebar Navigation** | Home · Recommend · Favourites · History · About |

---

## 🤖 AI Concepts Used

### 1. Content-Based Filtering
Content-Based Filtering recommends items similar to those a user has previously liked or is currently viewing. Instead of comparing users, it compares **items** based on their intrinsic attributes.

In CineMind AI, each movie is described by:
- **Genre** (weighted ×3 — most important)
- **Keywords / themes** (weighted ×2)
- **Description / synopsis** (weighted ×1)

### 2. TF-IDF Vectorization
**TF-IDF** stands for **Term Frequency – Inverse Document Frequency**.

```
TF(t,d)  = (number of times term t appears in document d) /
           (total number of terms in d)

IDF(t)   = log(total number of documents /
           number of documents containing t)

TF-IDF   = TF × IDF
```

- **TF** rewards terms that appear frequently in a specific movie's description.
- **IDF** penalises terms that appear in almost every movie (common words).
- The result is a sparse numerical vector where each dimension represents a unique term's weighted importance.

**Implementation in CineMind AI:**
```python
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),   # unigrams + bigrams
    max_features=8000,
    sublinear_tf=True,    # apply log(1+tf)
)
tfidf_matrix = vectorizer.fit_transform(movie_corpus)
```

### 3. Cosine Similarity
**Cosine Similarity** measures the cosine of the angle between two vectors in a multi-dimensional space:

```
cosine_similarity(A, B) = (A · B) / (||A|| × ||B||)
```

- **Score = 1.0** → Identical movies (same direction)
- **Score = 0.0** → Completely dissimilar (perpendicular vectors)
- **Score > 0.8** → Strongly similar (shown as green badge)
- **Score > 0.6** → Moderately similar (shown as amber badge)

```python
from sklearn.metrics.pairwise import cosine_similarity
sim_vector = cosine_similarity(query_vector, tfidf_matrix).flatten()
top_indices = sim_vector.argsort()[::-1][1:6]  # top 5, exclude self
```

---

## 🗂️ Project Structure

```
Task4_RecommendationSystem/
├── recommendation_system.py    # Main application (AI engine + full UI)
├── movies.csv                  # Curated dataset (120 movies)
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── favorites.json              # Auto-generated favourites store
├── history.json                # Auto-generated history store
├── screenshots/                # Application screenshots
└── assets/                     # Additional resources
```

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.10 or higher
- pip

### Step 1 – Clone / Download
```bash
git clone https://github.com/your-username/CineMind-AI.git
cd CineMind-AI
```

### Step 2 – Install Dependencies
```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install customtkinter pandas scikit-learn numpy
```

### Step 3 – Run
```bash
python recommendation_system.py
```

---

## 🖥️ Usage

1. **Launch** the application — a splash screen will animate while the AI model loads.
2. Navigate to **🎬 Recommend** in the sidebar.
3. **Type** a movie name in the search bar. Suggestions appear instantly.
4. **Select** a movie from the dropdown — its details appear in the card below.
5. Click **🧠 Recommend** — the AI generates **5 personalised recommendations** with:
   - Similarity percentage
   - Genre badge
   - AI explanation
   - Progress bar visualisation
6. **❤️ Favourite** any movie using the heart button.
7. **🎲 Surprise Me** for a random recommendation.
8. **📤 Export** the results to a text file.

---

## 📊 Dataset

The included `movies.csv` contains **120 curated movies** spanning genres:

| Genre | Example Movies |
|-------|---------------|
| Science Fiction | Interstellar, Dune, The Matrix, Arrival, Blade Runner 2049 |
| Action / Superhero | Avengers, The Dark Knight, Iron Man, John Wick |
| Drama | The Shawshank Redemption, Schindlers List, Parasite |
| Thriller / Horror | The Silence of the Lambs, Get Out, Hereditary |
| Fantasy | Lord of the Rings, Harry Potter, Dune |
| Animation | Toy Story, WALL-E, Up, The Lion King |
| Historical | Oppenheimer, 1917, Dunkirk, Hidden Figures |

**CSV Schema:**
```
movie_id, title, genre, keywords, description, year, rating, popularity
```

---

## 🎨 UI Design

| Element | Value |
|---------|-------|
| Background | `#0F1117` |
| Cards | `#1A1D29` with glassmorphism border |
| Accent | `#6C63FF` (purple) |
| Success | `#4CAF50` (green) |
| Warning | `#FFC107` (amber) |
| Font | Segoe UI (system) |
| Corners | 10–20px radius everywhere |
| Window | 1400 × 850, centered, resizable |

---

## 🔮 Future Enhancements

- [ ] **Collaborative Filtering** — user-based recommendation using rating data
- [ ] **Hybrid Model** — combine content-based + collaborative filtering
- [ ] **Movie Posters** — fetch and display real TMDB poster images
- [ ] **Neural Embeddings** — replace TF-IDF with sentence-transformers
- [ ] **User Profiles** — multi-user support with separate history/favourites
- [ ] **Online Dataset** — live sync from TMDB / IMDb API
- [ ] **Mood-Based Recommendations** — "I'm feeling adventurous" filter
- [ ] **Watch Providers** — show which streaming services host the film

---

## 👨‍💻 Developer

| | |
|--|--|
| **Name** | Sanjeevikumar D |
| **Internship** | CodSoft Artificial Intelligence Internship |
| **Task** | Task 4 – Movie Recommendation System |
| **Tech Stack** | Python · CustomTkinter · Pandas · Scikit-Learn |

---

## 📄 License

This project is developed for educational purposes as part of the **CodSoft AI Internship**. Feel free to use, modify, and learn from it.

---

<div align="center">

Built with ❤️ by **Sanjeevikumar D** · CodSoft AI Internship 2024

</div>
