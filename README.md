# 🎬 Movie Recommender System

[![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter&logoColor=white)](https://jupyter.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)

A **content-based movie recommendation engine** built from scratch using NLP and cosine similarity — with a fully deployed **Streamlit web application**. Input any movie title and the system returns the 5 most similar films with posters — powered by a 4,800+ movie dataset from TMDB.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Demo](#-demo)
- [Web Application](#-web-application)
- [How It Works](#-how-it-works)
- [Dataset](#-dataset)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Results](#-results)
- [Future Improvements](#-future-improvements)

---

## 🔍 Overview

This project implements a **Content-Based Filtering** recommender system that suggests movies based on their metadata — including genres, plot keywords, cast, crew, and overview. Unlike collaborative filtering, this approach requires no user interaction data, making it scalable and cold-start friendly.

**Key highlights:**
- Processes and merges two real-world TMDB datasets (movies + credits)
- Extracts structured features from nested JSON columns using `ast`
- Builds a rich "tags" representation per movie by combining 5 feature sources
- Vectorizes text using `CountVectorizer` (5,000 features, English stop words removed)
- Computes a **4,806 × 4,806 cosine similarity matrix** for instant lookup
- Serializes the model artifacts with `pickle` for deployment-ready use
- Deployed as an interactive **Streamlit web app** with movie poster fetching via TMDB API

---

## 🎥 Demo

```python
recommend('Batman')
# Output:
# Batman & Robin
# The Dark Knight Rises
# Batman Begins
# Batman Returns
# Batman Forever
```
Demo 1:
---
<img width="882" height="411" alt="screenshot1" src="https://github.com/user-attachments/assets/50314dfd-85d8-4b5e-a098-2ec4778d674e" />

Demo 2:
---
<img width="882" height="403" alt="screenshot2" src="https://github.com/user-attachments/assets/a598727a-e3f0-4b6e-b2d1-d1b60f547c40" />

Demo 3:
---
<img width="880" height="407" alt="screenshot3" src="https://github.com/user-attachments/assets/e3d9f8ae-c9c0-4b12-b664-fc4b0dbae7d1" />

---

## 🌐 Web Application

The project includes a fully functional **Streamlit web app** that brings the recommender to life with a clean UI and real movie posters.

**Features:**
- Searchable dropdown to select any movie from the dataset
- One-click **"Show Recommendation"** button
- Displays **5 recommended movies** with their posters fetched live from the TMDB API
- Dark-themed, responsive layout

### Run the App Locally

```bash
pip install streamlit requests
streamlit run app.py
```

```
Raw Data (movies.csv + credits.csv)
        │
        ▼
  Data Merging & Cleaning
  (merge on title, drop nulls & duplicates)
        │
        ▼
  Feature Extraction
  ┌─────────────────────────────────────────┐
  │  genres   → parse JSON → extract names  │
  │  keywords → parse JSON → extract names  │
  │  cast     → top 3 actors extracted      │
  │  crew     → director extracted          │
  │  overview → tokenized text              │
  └─────────────────────────────────────────┘
        │
        ▼
  Tag Construction
  (concatenate all features into one string per movie)
        │
        ▼
  CountVectorizer  →  5,000-dimensional Bag-of-Words vectors
        │
        ▼
  Cosine Similarity Matrix  (4806 × 4806)
        │
        ▼
  recommend(movie_title)  →  Top 5 Similar Movies
```

### Feature Engineering Details

| Feature | Processing |
|---|---|
| `genres` | Parsed from JSON, extracted name field |
| `keywords` | Parsed from JSON, extracted all keyword names |
| `cast` | Top 3 billed actors extracted from JSON |
| `crew` | Director identified by `job == 'Director'` |
| `overview` | Tokenized, lowercased, joined as string |

Multi-word names (e.g., `Sam Worthington`) were **collapsed to single tokens** (`SamWorthington`) to prevent the vectorizer from treating them as separate words — a subtle but important NLP design choice.

---

## 📦 Dataset

**Source:** [TMDB 5000 Movie Dataset – Kaggle](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)

| File | Rows | Key Columns |
|---|---|---|
| `movies.csv` | 4,809 | `title`, `genres`, `keywords`, `overview`, `budget`, `revenue` |
| `credits.csv` | 4,809 | `movie_id`, `title`, `cast`, `crew` |

After merging and cleaning: **4,806 movies** retained.

---

## 🛠 Tech Stack

| Library | Usage |
|---|---|
| `pandas` | Data loading, merging, cleaning, transformation |
| `numpy` | Numerical operations |
| `ast` | Safe parsing of stringified JSON columns |
| `scikit-learn` | `CountVectorizer`, `cosine_similarity` |
| `matplotlib` / `seaborn` | EDA visualizations |
| `pickle` | Model serialization for deployment |
| `streamlit` | Interactive web application frontend |
| `requests` | Fetching movie posters via TMDB API |

---

## 📁 Project Structure

```
movie-recommender-system/
│
├── Movie-Recommender-System.ipynb   # Main notebook (EDA + modeling)
├── app.py                           # Streamlit web application
├── movies.csv                       # Raw movies dataset
├── credits.csv                      # Raw credits dataset
├── movies.pkl                       # Serialized movie dataframe
├── similarity.pkl                   # Serialized cosine similarity matrix
├── screenshot1.png                  # App screenshot
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter streamlit requests
```

### Run the Notebook

```bash
git clone https://github.com/Mayank312003/movie-recommender-system.git
cd movie-recommender-system
jupyter notebook Movie-Recommender-System.ipynb
```

### Use the Recommender

```python
import pickle

movies = pickle.load(open('movies.pkl', 'rb'))
similarity = pickle.load(open('similarity.pkl', 'rb'))

def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    for i in movie_list:
        print(movies.iloc[i[0]].title)

recommend('The Dark Knight')
```

---

## 📊 Results

The model produces semantically meaningful recommendations by identifying movies that share genres, thematic keywords, cast, and directorial style — not just surface-level title similarity.

| Input Movie | Recommendations |
|---|---|
| Batman | Batman & Robin, The Dark Knight Rises, Batman Begins, Batman Returns, Batman Forever |
| Avatar | Similar sci-fi / action films with space/alien themes |

---

## 🔮 Future Improvements

- [ ] **TF-IDF Vectorization** — to downweight overly common terms
- [ ] **Hybrid filtering** — combine content-based with collaborative filtering using user ratings
- [x] **Streamlit Web App** — interactive UI with movie posters via TMDB API ✅
- [ ] **Word embeddings** — use Word2Vec or sentence transformers for richer semantic similarity

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you'd like to change.

---

## 👨‍💻 Author

**Mayank Yadav**

**Aspiring Data Scientist | Machine Learning Engineer**

> Built with ❤️ using Python, scikit-learn & Streamlit | Dataset courtesy of [TMDB](https://www.themoviedb.org/) via Kaggle
