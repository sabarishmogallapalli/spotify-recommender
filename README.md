# 🎵 Spotify Mood & Feature Clustering Engine
> An unsupervised machine learning pipeline that categorizes audio metadata using PCA and K-Means to deliver personalized, mood-based music recommendations via K-Nearest Neighbors.

---

## 🛠️ Badges & Tech Stack

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-Visualization-3776AB?style=for-the-badge)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Plotting-11557c?style=for-the-badge)

* **Live Demo:** [Run Locally via Python Script](#-local-setup-instructions)

---

## 💡 The Problem & Solution

**The Problem:** Traditional playlist recommendations often rely on broad genre categorization or general collaborative filtering. This approach frequently misses subtle audio nuances needed when a user wants a very specific emotional feel—such as a song that feels *"nostalgic yet optimistic"* (e.g., high acousticness, soft volume, moderate tempo, and high valence).

**The Solution:** This repository provides an end-to-end unsupervised learning pipeline that converts raw audio track metadata (danceability, energy, acousticness, valence, etc.) into structured cluster profiles. By using Principal Component Analysis (PCA) for dimensionality reduction and K-Means for clustering, the system maps custom target mood profiles into a vectorized space to surface identical cluster matches and top-5 nearest neighbor song recommendations using K-Nearest Neighbors (KNN).

---

## ⚙️ System Architecture & Data Flow

```text
[Raw Track Data] 
       │
       ▼
[Pre-processing & Categorical Encoding] (LabelEncoder)
       │
       ▼
[Feature Standardization] (StandardScaler)
       │
       ▼
[Dimensionality Reduction] (PCA Component Extraction)
       │
       ├─────────────────────────────────────────┐
       ▼                                         ▼
[Silhouette Score Evaluation]            [Correlation Heatmaps]
  (Determines Optimal K-Means K)          (Mood, Rhythm, Popularity)
       │
       ▼
[K-Means Mood Clustering]
       │
       ▼
[User Mood Vector Query] ──(Transform)──► [KNN Classification] ──► [Recommended Tracks]
```

### Key Workflow Components:
1. **Data Ingestion & Cleaning:** Extracts audio attributes (`bpm`, `nrgy`, `dnce`, `dB`, `live`, `val`, `dur`, `acous`, `spch`, `pop`) and encodes categorical genre columns.
2. **Exploratory Data Analysis:** Generates targeted correlation heatmaps for feature subsets (`mood_features`, `popularity_features`, `rhythm_features`).
3. **Dimensionality Reduction:** Compresses feature variance into Principal Components using `PCA`.
4. **Optimal Clustering:** Computes silhouette scores over candidate cluster counts to objectively select the best $k$ value for K-Means.
5. **Inference & Recommendation:** Passes simulated user mood inputs through the saved preprocessing transforms, predicts the assigned mood cluster, and retrieves the top-5 closest song matches via Euclidean distance using `KNeighborsClassifier`.

---

## 🚀 Local Setup Instructions

Follow these step-by-step commands to get the project running on your local machine:

### 1. Prerequisites
Ensure you have Python 3.8+ installed on your system.

### 2. Clone the Repository
```bash
git clone https://github.com/your-username/spotify-mood-clustering.git
cd spotify-mood-clustering
```

### 3. Set Up a Virtual Environment (Recommended)
```bash
# On macOS/Linux
python3 -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
venv\Scripts\activate
```

### 4. Install Dependencies
```bash
pip install pandas scikit-learn seaborn matplotlib
```

### 5. Prepare Dataset
Ensure your dataset file named `spotify_data.csv` is placed directly in the project root directory.

### 6. Run the Pipeline
```bash
python main.py
```