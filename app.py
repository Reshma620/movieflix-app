import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import plotly.graph_objects as go

# ─────────────────────────────────────────────
# PAGE CONFIG — 
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="MovieFlix",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# NETFLIX DARK THEME CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #141414;
        color: white;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #1a1a1a;
        border-right: 1px solid #333;
    }
    
    /* All text white */
    .stApp, .stApp p, .stApp label, .stApp span {
        color: #e5e5e5 !important;
    }
    
    /* Netflix red header */
    .netflix-header {
        background: linear-gradient(180deg, #000 0%, transparent 100%);
        padding: 20px 0;
        margin-bottom: 30px;
    }
    
    .netflix-logo {
        font-size: 48px;
        font-weight: 900;
        color: #e50914;
        letter-spacing: -2px;
        font-family: Arial Black, sans-serif;
    }
    
    /* Hero banner */
    .hero-banner {
        background: linear-gradient(to right, #000 30%, transparent 100%),
                    linear-gradient(to top, #141414 5%, transparent 50%);
        padding: 60px 40px;
        border-radius: 8px;
        margin-bottom: 40px;
        min-height: 300px;
        display: flex;
        flex-direction: column;
        justify-content: flex-end;
        background-color: #2a2a2a;
    }
    
    .hero-title {
        font-size: 52px;
        font-weight: 900;
        color: white;
        margin: 0;
        text-shadow: 2px 2px 8px rgba(0,0,0,0.8);
    }
    
    .hero-meta {
        font-size: 16px;
        color: #aaa;
        margin: 8px 0 16px 0;
    }
    
    .hero-badge {
        display: inline-block;
        background: #e50914;
        color: white;
        padding: 4px 12px;
        border-radius: 4px;
        font-size: 14px;
        font-weight: bold;
        margin-right: 8px;
    }
    
    /* Section titles (Trending Now, etc.) */
    .section-title {
        font-size: 24px;
        font-weight: 700;
        color: white;
        margin: 30px 0 15px 0;
        padding-left: 4px;
        border-left: 4px solid #e50914;
        padding-left: 12px;
    }
    
    /* Movie cards */
    .movie-card {
        background: #1e1e1e;
        border-radius: 8px;
        overflow: hidden;
        transition: transform 0.2s ease;
        cursor: pointer;
        border: 1px solid #333;
    }
    
    .movie-card:hover {
        transform: scale(1.05);
        border-color: #e50914;
    }
    
    .movie-title-card {
        font-size: 13px;
        font-weight: 600;
        color: white;
        padding: 8px 8px 2px 8px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    
    .movie-meta-card {
        font-size: 11px;
        color: #aaa;
        padding: 0 8px 8px 8px;
    }
    
    .rating-badge {
        color: #f5c518;
        font-weight: bold;
    }
    
    /* Input fields */
    .stTextInput input {
        background-color: #333 !important;
        color: white !important;
        border: 1px solid #555 !important;
        border-radius: 4px !important;
    }
    
    .stTextInput input:focus {
        border-color: #e50914 !important;
    }
    
    /* Selectbox */
    .stSelectbox select, [data-testid="stSelectbox"] {
        background-color: #333 !important;
        color: white !important;
    }
    
    /* Slider */
    .stSlider [data-testid="stThumbValue"] {
        color: #e50914 !important;
    }
    
    /* Buttons */
    .stButton button {
        background-color: #e50914 !important;
        color: white !important;
        border: none !important;
        border-radius: 4px !important;
        font-weight: 600 !important;
        padding: 8px 20px !important;
        transition: background 0.2s !important;
    }
    
    .stButton button:hover {
        background-color: #f40612 !important;
        transform: scale(1.02);
    }
    
    /* Divider */
    hr {
        border-color: #333 !important;
    }
    
    /* Metric cards */
    [data-testid="metric-container"] {
        background: #1e1e1e;
        border: 1px solid #333;
        border-radius: 8px;
        padding: 16px;
    }
    
    [data-testid="metric-container"] label {
        color: #aaa !important;
    }
    
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #e50914 !important;
        font-size: 28px !important;
    }
    
    /* Hide streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Watchlist tag */
    .watchlist-tag {
        background: #e50914;
        color: white;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# TMDB API — TO FETCH POSTER
# ─────────────────────────────────────────────
TMDB_API_KEY = "3a83522a642399a752a031ec240522b7"  

MOOD_EMOJIS = {
    "Happy": "😄",
    "Exciting": "🔥",
    "Emotional": "💔",
    "Dark": "🌑",
    "Family": "👨‍👩‍👧",
    "Neutral": "🎬"
}


@st.cache_data(ttl=86400)
def get_poster(title, year=None):
    """
     FROM TMDB poster fetch — 5 strategies USE FOR MAX COVERAGE:
    1. Clean title + year
    2. Clean title WITHOUT year
    3. BEFORE COLAN PART (e.g. "Lord of the Rings: ..." → "Lord of the Rings")
    4. FIRST FOUR WORDS
    5. FIRST TWO WORDS(last resort)
    """
    import re

    def fetch(query, yr=None):
        """Helper: search TMDB, return poster path url or none"""
        try:
            params = {"api_key": TMDB_API_KEY, "query": query, "language": "en-US", "include_adult": False}
            if yr and int(yr) > 1900:
                params["year"] = int(yr)
            resp = requests.get("https://api.themoviedb.org/3/search/movie", params=params, timeout=8)
            if resp.status_code != 200:
                return None
            results = resp.json().get("results", [])
            # first find exact title match
            for r in results:
                if r.get("poster_path") and query.lower() == r.get("title", "").lower():
                    return f"https://image.tmdb.org/t/p/w342{r['poster_path']}"
            # Phir partial match
            for r in results:
                if r.get("poster_path") and query.lower() in r.get("title", "").lower():
                    return f"https://image.tmdb.org/t/p/w342{r['poster_path']}"
            # if matche not found then first take the result
            if results and results[0].get("poster_path"):
                return f"https://image.tmdb.org/t/p/w342{results[0]['poster_path']}"
        except Exception:
            pass
        return None

    try:
        # Step 1: clean title
        clean = re.sub(r'\s*\(.*?\)\s*', '', str(title)).strip()  # "(2008)" remove
        clean = re.sub(r'\s*\[.*?\]\s*', '', clean).strip()        # "[2008]"  remove
        clean = clean.strip(" .,:-")                                      # trailing punctuation

        # Strategy 1: clean title + year
        result = fetch(clean, year)
        if result:
            return result

        # Strategy 2: clean ttitle without year
        result = fetch(clean)
        if result:
            return result

        # Strategy 3: before colan part
        # e.g. "Lord of the Rings: Return of the King" → "Lord of the Rings"
        if ":" in clean:
            before_colon = clean.split(":")[0].strip()
            if len(before_colon) > 3:
                result = fetch(before_colon, year)
                if result:
                    return result
                result = fetch(before_colon)
                if result:
                    return result

        # Strategy 4: first four words
        words = clean.split()
        if len(words) > 4:
            result = fetch(" ".join(words[:4]), year)
            if result:
                return result
            result = fetch(" ".join(words[:4]))
            if result:
                return result

        # Strategy 5: first two words( last resort)
        if len(words) > 2:
            result = fetch(" ".join(words[:2]))
            if result:
                return result

    except Exception:
        pass
    return None


@st.cache_data
def load_data():
    """LOAD movies.csv and prepare TF-idf for recommendations"""
    try:
        movies = pd.read_csv("movies.csv")
    except FileNotFoundError:
        st.error("❌ movies.csv NOT FOUND!Please upload the dataset to app directory.")
        st.stop()

    # Clean data
    movies["combined_features"] = movies["combined_features"].fillna("").astype(str)
    movies = movies[movies["combined_features"].str.strip() != ""].reset_index(drop=True)
    movies["IMDb_Rating"] = pd.to_numeric(movies["IMDb_Rating"], errors="coerce").fillna(0)
    movies["Year"] = pd.to_numeric(movies["Year"], errors="coerce").fillna(0).astype(int)
    movies["numVotes"] = pd.to_numeric(movies["numVotes"], errors="coerce").fillna(0)
    movies["runtimeMinutes"] = pd.to_numeric(movies["runtimeMinutes"], errors="coerce").fillna(0)

    # TF-IDF model
    tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
    tfidf_matrix = tfidf.fit_transform(movies["combined_features"])
    cosine_sim = cosine_similarity(tfidf_matrix)

    return movies, cosine_sim


def get_recommendations(title, movies, cosine_sim, n=10):
    """Ek movie ke liye similar movies dhundho"""
    matches = movies[movies["Title"].str.contains(title, case=False, na=False)]
    if matches.empty:
        return pd.DataFrame()
    idx = matches.index[0]
    scores = list(enumerate(cosine_sim[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:n+1]
    movie_indices = [i for i, _ in scores]
    return movies.iloc[movie_indices]


def show_movie_row(section_title, df, movies_df, cosine_sim, max_cols=5):
    """Ek horizontal movie row dikhao (Netflix style)"""
    if df.empty:
        return

    st.markdown(f'<div class="section-title">{section_title}</div>', unsafe_allow_html=True)
    
    top_movies = df.head(max_cols)
    cols = st.columns(max_cols)
    
    for i, (_, row) in enumerate(top_movies.iterrows()):
        with cols[i]:
            # Year pass karo — better poster matching ke liye
            year_val   = int(row["Year"]) if row["Year"] > 0 else None
            poster_url = get_poster(row["Title"], year=year_val)
            if poster_url:
                st.image(poster_url, use_container_width=True)
            else:
                mood  = row.get("Mood", "Neutral")
                emoji = MOOD_EMOJIS.get(mood, "🎬")
                st.markdown(f"""
                <div style="background:linear-gradient(135deg,#1a1a2e,#16213e);
                    height:200px;border-radius:6px;display:flex;
                    align-items:center;justify-content:center;
                    font-size:52px;border:1px solid #333;margin-bottom:4px;">
                    {emoji}
                </div>""", unsafe_allow_html=True)

            # Movie info
            title_short = row["Title"][:22] + "..." if len(row["Title"]) > 22 else row["Title"]
            year = int(row["Year"]) if row["Year"] > 0 else "N/A"
            rating = f"{row['IMDb_Rating']:.1f}" if row["IMDb_Rating"] > 0 else "N/A"
            
            st.markdown(f"""
            <div class="movie-title-card" title="{row['Title']}">{title_short}</div>
            <div class="movie-meta-card">
                <span class="rating-badge">⭐ {rating}</span> · {year}
            </div>
            """, unsafe_allow_html=True)

            # Watchlist button
            in_watchlist = row["Title"] in st.session_state.watchlist
            btn_label = "✓ Added" if in_watchlist else "+ My List"
            if st.button(btn_label, key=f"wl_{section_title}_{i}_{row['Title'][:10]}"):
                if in_watchlist:
                    st.session_state.watchlist.discard(row["Title"])
                else:
                    st.session_state.watchlist.add(row["Title"])
                st.rerun()


# ─────────────────────────────────────────────
# SESSION STATE — watchlist and ratings store
# ─────────────────────────────────────────────
if "watchlist" not in st.session_state:
    st.session_state.watchlist = set()

if "user_ratings" not in st.session_state:
    st.session_state.user_ratings = {}  # {title: rating}

if "watch_history" not in st.session_state:
    st.session_state.watch_history = []

if "active_tab" not in st.session_state:
    st.session_state.active_tab = "home"


# ─────────────────────────────────────────────
# DATA LOAD
# ─────────────────────────────────────────────
movies, cosine_sim = load_data()


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div style="font-size:32px;font-weight:900;color:#e50914;letter-spacing:-1px;">🎬 MovieFlix</div>', unsafe_allow_html=True)
    st.markdown("---")

    # Navigation
    st.markdown("**Navigate**")
    nav = st.radio(
        "",
        ["🏠 Home", "🔍 Search & Recommend", "❤️ My Watchlist", "📊 Analytics"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("**Filter Movies**")

    # Genre filter
    all_genres = sorted(movies["Genre"].dropna().unique().tolist())
    selected_genre = st.selectbox("Genre", ["All"] + all_genres)

    # Mood filter
    all_moods = sorted(movies["Mood"].dropna().unique().tolist())
    selected_mood = st.selectbox("Mood", ["All"] + all_moods)

    # Year range
    min_year = int(movies["Year"].min()) if movies["Year"].min() > 0 else 1900
    max_year = int(movies["Year"].max())
    year_range = st.slider("Year Range", min_year, max_year, (2000, max_year))

    # Rating filter
    min_rating = st.slider("Min IMDb Rating", 0.0, 10.0, 6.0, 0.5)

    st.markdown("---")
    st.markdown(f"**My List:** {len(st.session_state.watchlist)} movies")
    st.markdown(f"**Watched:** {len(st.session_state.watch_history)} movies")
    st.markdown(f"**Rated:** {len(st.session_state.user_ratings)} movies")


# ─────────────────────────────────────────────
# APPLY FILTERS
# ─────────────────────────────────────────────
filtered = movies.copy()

if selected_genre != "All":
    filtered = filtered[filtered["Genre"].str.contains(selected_genre, na=False)]

if selected_mood != "All":
    filtered = filtered[filtered["Mood"] == selected_mood]

filtered = filtered[
    (filtered["Year"] >= year_range[0]) &
    (filtered["Year"] <= year_range[1])
]
filtered = filtered[filtered["IMDb_Rating"] >= min_rating]


# ─────────────────────────────────────────────
# MAIN CONTENT BASED ON NAV
# ─────────────────────────────────────────────

# ── HOME PAGE ──
if "Home" in nav:
    # Netflix logo
    st.markdown('<div class="netflix-logo">MOVIEFLIX</div>', unsafe_allow_html=True)

    # Hero Banner — top rated movie
    if not filtered.empty:
        hero = filtered.nlargest(1, "IMDb_Rating").iloc[0]
        mood_emoji = MOOD_EMOJIS.get(hero.get("Mood", "Neutral"), "🎬")
        runtime = f"{int(hero['runtimeMinutes'])} min" if hero["runtimeMinutes"] > 0 else ""
        
        st.markdown(f"""
        <div class="hero-banner">
            <div style="font-size:14px; color:#e50914; font-weight:600; margin-bottom:8px;">
                {mood_emoji} FEATURED TODAY
            </div>
            <div class="hero-title">{hero['Title']}</div>
            <div class="hero-meta">
                ⭐ {hero['IMDb_Rating']:.1f} &nbsp;·&nbsp; 
                {int(hero['Year'])} &nbsp;·&nbsp; 
                {hero.get('Genre','')[:30]} &nbsp;·&nbsp;
                {runtime}
            </div>
            <div>
                <span class="hero-badge">{hero.get('Mood','')}</span>
                <span style="color:#aaa; font-size:13px;">
                    {int(hero['numVotes']):,} votes
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Stats row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Movies", f"{len(filtered):,}")
    with col2:
        st.metric("Avg Rating", f"{filtered['IMDb_Rating'].mean():.1f} ⭐")
    with col3:
        st.metric("Genres", filtered["Genre"].nunique())
    with col4:
        st.metric("Year Range", f"{int(filtered['Year'].min())}–{int(filtered['Year'].max())}")

    st.markdown("---")

    # Trending — most votes
    trending = filtered.nlargest(5, "numVotes")
    show_movie_row("🔥 Trending Now", trending, movies, cosine_sim)

    # Top Rated
    top_rated = filtered.nlargest(5, "IMDb_Rating")
    show_movie_row("⭐ Top Rated", top_rated, movies, cosine_sim)

    # By Mood rows
    for mood, emoji in MOOD_EMOJIS.items():
        mood_df = filtered[filtered["Mood"] == mood].nlargest(5, "IMDb_Rating")
        if len(mood_df) >= 3:
            show_movie_row(f"{emoji} {mood} Picks", mood_df, movies, cosine_sim)

    # Recent releases
    recent = filtered[filtered["Year"] >= 2020].nlargest(5, "IMDb_Rating")
    if not recent.empty:
        show_movie_row("🆕 New Releases (2020+)", recent, movies, cosine_sim)


# ── SEARCH & RECOMMEND PAGE ──
elif "Search" in nav:
    st.markdown('<div class="section-title">🔍 Search & Get Recommendations</div>', unsafe_allow_html=True)

    search_query = st.text_input("", placeholder="Movie naam likho... (e.g. Inception, Avatar)")

    col1, col2 = st.columns([1, 1])

    with col1:
        # Search results
        if search_query:
            results = movies[movies["Title"].str.contains(search_query, case=False, na=False)]
            
            if results.empty:
                st.warning(f"'{search_query}' nahi mila. Koi aur naam try karo.")
            else:
                st.markdown(f"**{len(results)} movies mili:**")
                
                for _, row in results.head(8).iterrows():
                    with st.container():
                        c1, c2 = st.columns([3, 1])
                        with c1:
                            st.markdown(f"**{row['Title']}** ({int(row['Year'])})")
                            st.markdown(f"⭐ {row['IMDb_Rating']:.1f} · {row.get('Genre','')[:40]} · {row.get('Mood','')}")
                        with c2:
                            if st.button("Recommend", key=f"rec_{row['Title'][:15]}"):
                                st.session_state["selected_movie"] = row["Title"]
                        st.markdown("---")

    with col2:
        # Recommendations
        selected = st.session_state.get("selected_movie", "")
        
        if search_query and not selected:
            # Auto recommend on first result
            first_match = movies[movies["Title"].str.contains(search_query, case=False, na=False)]
            if not first_match.empty:
                selected = first_match.iloc[0]["Title"]

        if selected:
            st.markdown(f"**Because you liked '{selected}':**")
            recs = get_recommendations(selected, movies, cosine_sim, n=8)
            
            if not recs.empty:
                for _, row in recs.iterrows():
                    mood_e = MOOD_EMOJIS.get(row.get("Mood", "Neutral"), "🎬")
                    in_wl = row["Title"] in st.session_state.watchlist
                    
                    st.markdown(f"""
                    <div style="background:#1e1e1e; border-radius:6px; padding:10px; margin:6px 0; border:1px solid #333;">
                        <div style="font-weight:600; color:white;">{mood_e} {row['Title']}</div>
                        <div style="color:#aaa; font-size:12px;">
                            ⭐ {row['IMDb_Rating']:.1f} · {int(row['Year'])} · {row.get('Genre','')[:30]}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    btn_txt = "✓ In My List" if in_wl else "+ My List"
                    if st.button(btn_txt, key=f"rec_wl_{row['Title'][:15]}"):
                        if in_wl:
                            st.session_state.watchlist.discard(row["Title"])
                        else:
                            st.session_state.watchlist.add(row["Title"])
                        st.rerun()

    # Rate a movie section
    st.markdown("---")
    st.markdown('<div class="section-title">⭐ Rate a Movie</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        rate_movie = st.selectbox("Movie chunoo", movies["Title"].tolist(), key="rate_sel")
    with col2:
        user_rating = st.slider("Rating", 1, 10, 7, key="rate_slider")
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Save Rating"):
            st.session_state.user_ratings[rate_movie] = user_rating
            if rate_movie not in st.session_state.watch_history:
                st.session_state.watch_history.append(rate_movie)
            st.success(f"✅ '{rate_movie}' ko {user_rating}/10 rating di!")


# ── WATCHLIST PAGE ──
elif "Watchlist" in nav:
    st.markdown('<div class="section-title">❤️ My Watchlist</div>', unsafe_allow_html=True)

    if not st.session_state.watchlist:
        st.markdown("""
        <div style="text-align:center; padding:80px 0; color:#555;">
            <div style="font-size:64px;">🎬</div>
            <div style="font-size:20px; margin-top:16px;">Tumhari list khali hai!</div>
            <div style="font-size:14px; margin-top:8px; color:#444;">
                Home ya Search mein "+ My List" button dabao
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        wl_movies = movies[movies["Title"].isin(st.session_state.watchlist)]
        
        st.markdown(f"**{len(wl_movies)} movies tumhari list mein hain**")
        st.markdown("---")
        
        cols = st.columns(4)
        for i, (_, row) in enumerate(wl_movies.iterrows()):
            with cols[i % 4]:
                year_val = int(row["Year"]) if row["Year"] > 0 else None
                poster = get_poster(row["Title"], year=year_val)
                if poster:
                    st.image(poster, use_container_width=True)
                else:
                    mood = row.get("Mood", "Neutral")
                    emoji = MOOD_EMOJIS.get(mood, "🎬")
                    st.markdown(f"""
                    <div style="background:#1e1e2e; height:140px; border-radius:6px;
                                display:flex; align-items:center; justify-content:center;
                                font-size:40px; border:1px solid #333; margin-bottom:6px;">
                        {emoji}
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown(f"**{row['Title'][:20]}**")
                rating_display = st.session_state.user_ratings.get(row["Title"], None)
                if rating_display:
                    st.markdown(f"Your rating: ⭐ {rating_display}/10")
                else:
                    st.markdown(f"⭐ {row['IMDb_Rating']:.1f} IMDb")
                
                if st.button("Remove", key=f"rm_{row['Title'][:15]}"):
                    st.session_state.watchlist.discard(row["Title"])
                    st.rerun()

    # Watch history
    if st.session_state.watch_history:
        st.markdown("---")
        st.markdown('<div class="section-title">📺 Watch History</div>', unsafe_allow_html=True)
        
        hist_movies = movies[movies["Title"].isin(st.session_state.watch_history)]
        show_movie_row("Recently Watched", hist_movies, movies, cosine_sim)


# ── ANALYTICS PAGE ──
elif "Analytics" in nav:
    st.markdown('<div class="section-title">📊 Movie Analytics</div>', unsafe_allow_html=True)

    # Top stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Movies", f"{len(filtered):,}")
    with col2:
        st.metric("Avg IMDb Rating", f"{filtered['IMDb_Rating'].mean():.2f}")
    with col3:
        st.metric("Most Common Mood", filtered["Mood"].mode().iloc[0] if not filtered.empty else "N/A")
    with col4:
        best = filtered.nlargest(1, "IMDb_Rating")
        st.metric("Highest Rated", best.iloc[0]["Title"][:20] if not best.empty else "N/A")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        # Top genres bar chart
        genre_counts = filtered["Genre"].str.split(",").explode().str.strip().value_counts().head(10)
        fig1 = px.bar(
            x=genre_counts.values,
            y=genre_counts.index,
            orientation="h",
            title="Top 10 Genres",
            color=genre_counts.values,
            color_continuous_scale="reds",
        )
        fig1.update_layout(
            plot_bgcolor="#1e1e1e",
            paper_bgcolor="#141414",
            font_color="white",
            title_font_color="#e50914",
            showlegend=False,
            coloraxis_showscale=False,
            yaxis=dict(autorange="reversed")
        )
        fig1.update_traces(marker_line_width=0)
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        # Mood pie chart
        mood_counts = filtered["Mood"].value_counts()
        fig2 = px.pie(
            values=mood_counts.values,
            names=mood_counts.index,
            title="Movies by Mood",
            color_discrete_sequence=px.colors.sequential.RdBu
        )
        fig2.update_layout(
            plot_bgcolor="#1e1e1e",
            paper_bgcolor="#141414",
            font_color="white",
            title_font_color="#e50914",
        )
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        # IMDb rating distribution
        fig3 = px.histogram(
            filtered,
            x="IMDb_Rating",
            nbins=20,
            title="IMDb Rating Distribution",
            color_discrete_sequence=["#e50914"]
        )
        fig3.update_layout(
            plot_bgcolor="#1e1e1e",
            paper_bgcolor="#141414",
            font_color="white",
            title_font_color="#e50914",
            bargap=0.05
        )
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        # Movies per year
        year_counts = filtered[filtered["Year"] > 1950].groupby("Year").size().reset_index(name="count")
        fig4 = px.line(
            year_counts,
            x="Year",
            y="count",
            title="Movies Released Per Year",
            color_discrete_sequence=["#e50914"]
        )
        fig4.update_layout(
            plot_bgcolor="#1e1e1e",
            paper_bgcolor="#141414",
            font_color="white",
            title_font_color="#e50914",
        )
        fig4.update_traces(line_width=2)
        st.plotly_chart(fig4, use_container_width=True)

    # Top 10 movies table
    st.markdown("---")
    st.markdown('<div class="section-title">🏆 Top 10 Movies (Filtered)</div>', unsafe_allow_html=True)
    
    top10 = filtered.nlargest(10, "IMDb_Rating")[
        ["Title", "Genre", "Year", "IMDb_Rating", "Mood", "numVotes"]
    ].copy()
    top10["numVotes"] = top10["numVotes"].apply(lambda x: f"{int(x):,}")
    top10.columns = ["Title", "Genre", "Year", "IMDb Rating", "Mood", "Votes"]
    top10.index = range(1, 11)
    
    st.dataframe(
        top10,
        use_container_width=True,
        height=380
    )

    # User ratings chart
    if st.session_state.user_ratings:
        st.markdown("---")
        st.markdown('<div class="section-title">⭐ Your Ratings</div>', unsafe_allow_html=True)
        
        ur_df = pd.DataFrame(
            list(st.session_state.user_ratings.items()),
            columns=["Movie", "Your Rating"]
        )
        fig5 = px.bar(
            ur_df,
            x="Movie",
            y="Your Rating",
            title="Movies You've Rated",
            color="Your Rating",
            color_continuous_scale="reds",
        )
        fig5.update_layout(
            plot_bgcolor="#1e1e1e",
            paper_bgcolor="#141414",
            font_color="white",
            title_font_color="#e50914",
            coloraxis_showscale=False
        )
        st.plotly_chart(fig5, use_container_width=True)