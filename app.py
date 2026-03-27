import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="CineMatch Recommender", layout="wide")
st.title("🎬 CineMatch: Smart Movie Recommender")

# --- DATA LOADING & CACHING ---
@st.cache_data
def load_data():
    # Load the data generated from Phase 1 & 2
    df = pd.read_csv('tmdb_movies_processed.csv')
    df = df.fillna('')
    # Extract Release Year for temporal filtering
    df['release_year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year
    df['release_year'] = df['release_year'].fillna(2000).astype(int)
    return df

df_movies = load_data()

# Compute Cosine Similarity once on load
@st.cache_resource
def compute_similarity(df):
    count = CountVectorizer(stop_words='english')
    count_matrix = count.fit_transform(df['soup'])
    return cosine_similarity(count_matrix, count_matrix)

cosine_sim = compute_similarity(df_movies)
indices = pd.Series(df_movies.index, index=df_movies['title']).drop_duplicates()

# --- SIDEBAR: CORE FILTERING CAPABILITIES ---
st.sidebar.header("Filter Database")

# 1. Temporal Filter
min_year, max_year = int(df_movies['release_year'].min()), int(df_movies['release_year'].max())
selected_years = st.sidebar.slider("Release Period", min_value=min_year, max_value=max_year, value=(2010, max_year))

# 2. Quality Thresholds
min_rating = st.sidebar.slider("Minimum Rating", 0.0, 10.0, 6.0)

# 3. Content Specs (Runtime)
max_runtime = st.sidebar.slider("Maximum Runtime (mins)", 60, 240, 180)

# Apply baseline filters to a working dataframe
filtered_df = df_movies[
    (df_movies['release_year'] >= selected_years[0]) & 
    (df_movies['release_year'] <= selected_years[1]) &
    (df_movies['vote_average'] >= min_rating) &
    (df_movies['runtime'] <= max_runtime)
]

# 4. Genre Combinations (Dynamic based on remaining movies)
all_genres = set(", ".join(filtered_df['genres'].tolist()).split(", "))
selected_genres = st.sidebar.multiselect("Select Genres", list(all_genres))

if selected_genres:
    # Filter movies that contain ANY of the selected genres
    pattern = '|'.join(selected_genres)
    filtered_df = filtered_df[filtered_df['genres'].str.contains(pattern, case=False, na=False)]

st.sidebar.write(f"**Movies matching filters: {len(filtered_df)}**")


# --- MAIN LAYOUT: TABS ---
tab1, tab2, tab3 = st.tabs(["Find Similar Movies", "Mood Matcher (NLP)", "Data Insights"])

# TAB 1: Content-Based Filtering + Explanations
with tab1:
    st.subheader("Discover Based on Your Favorites")
    selected_movie = st.selectbox("Type or select a movie you like:", df_movies['title'].tolist())
    
    if st.button("Recommend Movies"):
        idx = indices[selected_movie]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6] # Top 5
        movie_indices = [i[0] for i in sim_scores]
        
        recommendations = df_movies.iloc[movie_indices]
        
        cols = st.columns(5)
        for i, col in enumerate(cols):
            movie = recommendations.iloc[i]
            with col:
                st.write(f"**{movie['title']}** ({movie['release_year']})")
                st.caption(f"⭐ {movie['vote_average']}")
                # Enhanced Feature 1: Similarity Explanation
                st.info(f"Why? Shares genres like {movie['genres'].split(',')[0]} and director {movie['director']}.")

# TAB 2: NLP Sentiment Matching
with tab2:
    st.subheader("Find Movies by Mood")
    mood = st.radio("What kind of vibe are you looking for?", ["Uplifting & Positive", "Dark & Gritty"])
    
    if st.button("Match Mood"):
        # We apply the user's sidebar filters (filtered_df) to the mood matcher!
        if mood == "Uplifting & Positive":
            mood_df = filtered_df.sort_values(by='sentiment_score', ascending=False).head(5)
        else:
            mood_df = filtered_df.sort_values(by='sentiment_score', ascending=True).head(5)
            
        for _, row in mood_df.iterrows():
            st.markdown(f"### {row['title']} ({row['release_year']})")
            st.write(f"**Plot:** {row['overview']}")
            st.write("---")

# TAB 3: Data Visualizations (Enhanced Feature 2)
with tab3:
    st.subheader("Dataset Trends")
    # Show how ratings trend over the years
    fig = px.scatter(
        filtered_df, 
        x="release_year", 
        y="vote_average", 
        color="vote_average",
        hover_data=['title', 'director'],
        title="Movie Ratings over Time (Based on Current Filters)"
    )
    st.plotly_chart(fig, use_container_width=True)