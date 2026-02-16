"""
Resume Ranking Web App
======================
A Streamlit application that ranks PDF resumes against a job description
using TF-IDF vectorization and Cosine Similarity.

How to run
----------
1.  Install dependencies (once):
        pip install streamlit pandas scikit-learn pypdf nltk

2.  Launch the app:
        streamlit run app.py

3.  Open the URL shown in your terminal (usually http://localhost:8501).
"""

# ── Imports ─────────────────────────────────────────────────────────────
import re
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import nltk
from nltk.corpus import stopwords

# ── NLTK bootstrap (runs once, cached) ─────────────────────────────────
@st.cache_resource
def _download_nltk_data():
    """Download required NLTK data files on first launch."""
    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)
    nltk.download("averaged_perceptron_tagger", quiet=True)

_download_nltk_data()

# Try to load the lemmatizer (graceful fallback if unavailable)
try:
    from nltk.stem import WordNetLemmatizer
    LEMMATIZER = WordNetLemmatizer()
    HAS_LEMMATIZER = True
except Exception:
    HAS_LEMMATIZER = False

# ── Domain-aware stopwords & tech-keyword map ──────────────────────────
DOMAIN_STOPWORDS = set(stopwords.words("english")) - {
    "python", "java", "sql", "machine", "learning", "data", "experience",
    "development", "engineer", "design", "management", "skill", "work",
}

TECH_MAP = {
    "c++": "cplusplus",
    "c#": "csharp",
    ".net": "dotnet",
    "node.js": "nodejs",
    "asp.net": "aspnet",
}


# ── Helper functions (from your notebook) ──────────────────────────────
def clean_text(text: str, use_lemmatization: bool = True) -> str:
    """
    Clean and preprocess text with tech-keyword preservation.

    - Preserves C++, C#, .NET, Node.js, etc.
    - Keeps important 2-letter acronyms (AI, ML, UI, UX, QA, IT, CV)
    - Applies lemmatization for better matching
    - Uses domain-aware stopword filtering
    """
    if not text or (isinstance(text, float) and pd.isna(text)):
        return ""

    text = str(text).lower()

    # Preserve technical terms before cleaning
    for tech, placeholder in TECH_MAP.items():
        text = text.replace(tech, placeholder)

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    # Remove email addresses
    text = re.sub(r"\S+@\S+", "", text)
    # Remove special characters, keep alphanumerics
    text = re.sub(r"[^a-z0-9\s]", "", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Tokenize, filter, and lemmatize
    tokens = text.split()
    cleaned_tokens = []
    for token in tokens:
        if len(token) >= 3 or token in {"ai", "ml", "ui", "ux", "qa", "it", "cv"}:
            if token not in DOMAIN_STOPWORDS:
                if use_lemmatization and HAS_LEMMATIZER:
                    token = LEMMATIZER.lemmatize(token, pos="v")
                cleaned_tokens.append(token)

    return " ".join(cleaned_tokens)


def extract_text_from_pdf(pdf_file) -> str | None:
    """
    Extract text from an uploaded PDF file object.

    Returns the concatenated page text, or None on failure.
    """
    try:
        reader = PdfReader(pdf_file)
        pages_text = []
        for page in reader.pages:
            try:
                page_text = page.extract_text()
                if page_text:
                    pages_text.append(page_text)
            except Exception:
                continue  # skip corrupt pages

        return "\n".join(pages_text) if pages_text else None
    except Exception:
        return None


def rank_resumes(
    job_description: str,
    filenames: list[str],
    resume_texts: list[str],
    max_features: int = 5000,
    ngram_range: tuple = (1, 2),
    min_df: int = 2,
    max_df: float = 0.95,
) -> tuple[pd.DataFrame, TfidfVectorizer, list[str]]:
    """
    Rank resumes against a job description using TF-IDF + Cosine Similarity.

    Returns a tuple of (DataFrame, vectorizer, cleaned_resumes).
    """
    cleaned_jd = clean_text(job_description)
    cleaned_resumes = [clean_text(t) for t in resume_texts]

    all_documents = [cleaned_jd] + cleaned_resumes

    # Adapt min_df for very small uploads
    effective_min_df = min(min_df, len(all_documents))

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=effective_min_df,
        max_df=max_df,
    )
    tfidf_matrix = vectorizer.fit_transform(all_documents)

    # Cosine similarity: JD vector vs. every resume vector
    jd_vector = tfidf_matrix[0:1]
    resume_vectors = tfidf_matrix[1:]
    scores = cosine_similarity(jd_vector, resume_vectors).flatten() * 100

    results = (
        pd.DataFrame({"Filename": filenames, "Fit Score (%)": scores})
        .sort_values("Fit Score (%)", ascending=False)
        .reset_index(drop=True)
    )
    results.insert(0, "Rank", range(1, len(results) + 1))
    return results, vectorizer, cleaned_resumes


def extract_top_keywords(cleaned_text: str, vectorizer: TfidfVectorizer, top_n: int = 10) -> list[str]:
    """
    Extract top keywords from cleaned resume text using TF-IDF weights.
    
    Returns list of top N keywords sorted by importance.
    """
    try:
        # Transform the text
        text_vector = vectorizer.transform([cleaned_text])
        
        # Get feature names and their scores
        feature_names = vectorizer.get_feature_names_out()
        scores = text_vector.toarray()[0]
        
        # Get top N features by score
        top_indices = scores.argsort()[-top_n:][::-1]
        top_keywords = [feature_names[i] for i in top_indices if scores[i] > 0]
        
        return top_keywords[:top_n]
    except Exception:
        return []


# ── Streamlit page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="CVision - AI Resume Analyzer",
    page_icon="�",
    layout="centered",
)

# ── Styling: highlight the top candidate row in green ──────────────────
def highlight_top(row):
    """Apply green background to the rank-1 row."""
    if row["Rank"] == 1:
        return ["background-color: #d4edda; font-weight: bold"] * len(row)
    return [""] * len(row)


# ── UI ─────────────────────────────────────────────────────────────────
st.markdown("# 📄 <span style='color: red;'>CVision</span>", unsafe_allow_html=True)
st.markdown("### AI-Powered Resume Intelligence")
st.markdown(
    "Paste a **Job Description**, upload **PDF resumes**, and click "
    "**Analyze Resumes** to see how each candidate matches the role."
)

st.divider()

# --- Input section ---
job_description = st.text_area(
    "📋 Job Description",
    height=250,
    placeholder="Paste the full job description here…",
)

uploaded_files = st.file_uploader(
    "📎 Upload PDF Resumes",
    type=["pdf"],
    accept_multiple_files=True,
    help="Select one or more PDF files.",
)

st.divider()

# --- Process & output ---
rank_button = st.button("🚀 Analyze Resumes", type="primary", use_container_width=True)

if rank_button:
    # ── Validation ──────────────────────────────────────────────────
    if not job_description.strip():
        st.error("Please paste a Job Description before ranking.")
        st.stop()
    if not uploaded_files:
        st.error("Please upload at least one PDF resume.")
        st.stop()

    # ── Extract text from each PDF ──────────────────────────────────
    filenames: list[str] = []
    resume_texts: list[str] = []
    skipped: list[str] = []

    with st.spinner("Extracting text from PDFs…"):
        for pdf_file in uploaded_files:
            text = extract_text_from_pdf(pdf_file)
            if text and text.strip():
                filenames.append(pdf_file.name)
                resume_texts.append(text)
            else:
                skipped.append(pdf_file.name)

    if skipped:
        st.warning(
            f"⚠️ Could not extract text from {len(skipped)} file(s): "
            f"{', '.join(skipped)}"
        )

    if not filenames:
        st.error("No readable resumes found. Please check your PDF files.")
        st.stop()

    # ── Rank ────────────────────────────────────────────────────────
    with st.spinner("Ranking resumes…"):
        results_df, vectorizer, cleaned_resumes = rank_resumes(job_description, filenames, resume_texts)

    # ── Display results ─────────────────────────────────────────────
    st.success(f"✅ Analyzed **{len(results_df)}** resume(s)")

    # Score summary
    col1, col2, col3 = st.columns(3)
    col1.metric("🥇 Top Score", f"{results_df['Fit Score (%)'].iloc[0]:.1f}%")
    col2.metric("📊 Average", f"{results_df['Fit Score (%)'].mean():.1f}%")
    col3.metric("📉 Lowest", f"{results_df['Fit Score (%)'].iloc[-1]:.1f}%")

    st.markdown("### 🏆 Ranked Results")

    # Styled table with green highlight on top candidate
    styled = (
        results_df.style
        .apply(highlight_top, axis=1)
        .format({"Fit Score (%)": "{:.2f}%"})
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)

    # Bar chart - sorted from highest to lowest
    st.markdown("### 📊 Fit Score Distribution")
    chart_df = results_df.set_index("Filename")[["Fit Score (%)"]]
    st.bar_chart(chart_df, horizontal=True)
    
    # Keywords section
    st.markdown("### 🔑 Key Skills & Qualifications by Candidate")
    st.caption("Matched keywords between job description and resume")
    
    # Create a mapping of original order to match cleaned_resumes
    filename_to_cleaned = dict(zip(filenames, cleaned_resumes))
    
    for _, row in results_df.iterrows():
        rank = row["Rank"]
        filename = row["Filename"]
        
        # Get the cleaned text for this file
        cleaned_text = filename_to_cleaned[filename]
        
        # Extract top keywords
        keywords = extract_top_keywords(cleaned_text, vectorizer, top_n=8)
        
        # Display directly without dropdown
        emoji = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "📄"
        st.markdown(f"**{emoji} Rank #{rank}: {filename}**")
        if keywords:
            # Display as tags
            keywords_html = " ".join([f'<span style="background-color: #e1f5ff; padding: 4px 12px; border-radius: 12px; margin: 2px; display: inline-block; font-size: 14px;">{kw}</span>' for kw in keywords])
            st.markdown(keywords_html, unsafe_allow_html=True)
        else:
            st.info("No significant keywords detected")
        st.markdown("")

# ── Footer ─────────────────────────────────────────────────────────────
st.divider()
st.caption("Built with Streamlit · TF-IDF + Cosine Similarity · NLTK")
