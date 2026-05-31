"""
data_processing.py
──────────────────
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

print("📂 Step 1: Loading raw data...")

# CSV FILE NAME
RAW_CSV = "results_with_crew.csv"  

if not os.path.exists(RAW_CSV):
    print(f"❌ '{RAW_CSV}' NOT FOUND!")
    print("PUT CSV FOLDER HERE")
    exit(1)

movies_raw = pd.read_csv(RAW_CSV)
print(f"✅ Loaded: {movies_raw.shape[0]} rows, {movies_raw.shape[1]} columns")
print("Columns:", movies_raw.columns.tolist())

print("\n📂 Step 2: Handling missing values...")
movies_raw['genres']    = movies_raw['genres'].fillna('Unknown')
movies_raw['directors'] = movies_raw['directors'].fillna('Unknown')
movies_raw['writers']   = movies_raw['writers'].fillna('Unknown')
print("✅ Missing values filled")

print("\n📂 Step 3: Creating Mood column...")
def detect_mood(genre):
    if "Comedy" in str(genre):
        return "Happy"
    elif "Action" in str(genre) or "Adventure" in str(genre):
        return "Exciting"
    elif "Drama" in str(genre) or "Romance" in str(genre):
        return "Emotional"
    elif "Horror" in str(genre) or "Thriller" in str(genre):
        return "Dark"
    elif "Animation" in str(genre) or "Family" in str(genre):
        return "Family"
    else:
        return "Neutral"

movies_raw["Mood"] = movies_raw["genres"].apply(detect_mood)
print("✅ Mood column created")
print(movies_raw["Mood"].value_counts())

print("\n📂 Step 4: Creating combined_features...")
movies_raw["combined_features"] = (
    movies_raw["genres"].astype(str)    + " " +
    movies_raw["directors"].astype(str) + " " +
    movies_raw["writers"].astype(str)   + " " +
    movies_raw["Mood"].astype(str)
)
print("✅ combined_features created")

print("\n📂 Step 5: Calculating popularity scores...")
movies_raw["Popularity_Score"] = movies_raw["averageRating"] * movies_raw["numVotes"]
movies_raw["MostPopular"]      = movies_raw["Popularity_Score"].rank(ascending=False)
movies_raw["MostWatched"]      = movies_raw["numVotes"].rank(ascending=False)
print("✅ Popularity scores created")

print("\n📂 Step 6: Selecting and renaming columns...")
# COLUMNS NAME CHANGE
movies_final = movies_raw[[
    "primaryTitle",
    "genres",
    "startYear",
    "averageRating",
    "numVotes",
    "runtimeMinutes",
    "Mood",
    "combined_features"
]].copy()

movies_final.rename(columns={
    "primaryTitle":  "Title",
    "genres":        "Genre",
    "startYear":     "Year",
    "averageRating": "IMDb_Rating"
}, inplace=True)

print("\n📂 Step 7: Cleaning data...")
movies_final["combined_features"] = movies_final["combined_features"].fillna("").astype(str)
movies_final = movies_final[movies_final["combined_features"].str.strip() != ""]
movies_final = movies_final.dropna(subset=["Title", "IMDb_Rating"])
movies_final = movies_final.reset_index(drop=True)
print(f"✅ Clean rows: {len(movies_final)}")

print("\n📂 Step 8: Saving movies.csv...")
movies_final.to_csv("movies.csv", index=False)
print("✅ movies.csv saved!")

print("\n📂 Step 9: Testing TF-IDF model...")
tfidf        = TfidfVectorizer(stop_words="english", max_features=5000)
tfidf_matrix = tfidf.fit_transform(movies_final["combined_features"])
cosine_sim   = cosine_similarity(tfidf_matrix)
print(f"✅ TF-IDF matrix shape: {tfidf_matrix.shape}")
print(f"✅ Cosine similarity matrix: {cosine_sim.shape}")

print("\n📂 Step 10: Quick recommendation test...")
def test_recommend(title, n=3):
    matches = movies_final[movies_final["Title"].str.contains(title, case=False, na=False)]
    if matches.empty:
        print(f"  '{title}' not found in dataset")
        return
    idx    = matches.index[0]
    scores = sorted(enumerate(cosine_sim[idx]), key=lambda x: x[1], reverse=True)[1:n+1]
    print(f"  Similar to '{movies_final.iloc[idx]['Title']}':")
    for i, (midx, score) in enumerate(scores, 1):
        r = movies_final.iloc[midx]
        print(f"    {i}. {r['Title']} ({int(r['Year'])}) ⭐{r['IMDb_Rating']}")

# TEST MOVIE FIRST
first_title = movies_final.iloc[0]["Title"]
test_recommend(first_title)

print("\n" + "="*50)
print("🎉 ALL DONE! RUN THIS COMMAND")
print("="*50)
print("  streamlit run app.py")
print("="*50)