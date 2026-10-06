import re
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

st.set_page_config(page_title="ReviewCheck AI", page_icon="🛍️", layout="centered")
st.title("🛍️ AI Product Review Checker — Any Product")
st.write("Train a text classifier using labeled examples from any product category, then check reviews for any product. Predictions are estimates, not proof of fraud.")

@st.cache_resource
def train_model(texts, labels):
    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=30000, strip_accents="unicode", sublinear_tf=True)),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    model.fit(texts, labels)
    return model

def normalize_label(value):
    value = str(value).strip().lower()
    if value in {"fake", "deceptive", "spam", "1", "true", "yes", "suspicious"}:
        return "Fake"
    if value in {"genuine", "real", "truthful", "0", "false", "no", "authentic"}:
        return "Genuine"
    return None

uploaded = st.file_uploader("Upload labeled training data (CSV, up to 200 MB)", type=["csv"], help="One CSV file up to 200 MB. Include review text and a label column. Data can cover any product category. Labels can say Fake/Genuine or 1/0.")
model = None
if uploaded:
    try:
        data = None
        for encoding in ("utf-8-sig", "cp1252", "latin-1", "utf-16"):
            try:
                uploaded.seek(0)
                data = pd.read_csv(uploaded, encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        if data is None:
            st.error("Could not decode this CSV. In Excel, use Save As and choose CSV UTF-8 (Comma delimited), then upload it again.")
            st.stop()
        st.caption("CSV columns: " + ", ".join(map(str, data.columns)))
        text_col = st.selectbox("Select the review text column", data.columns)
        label_col = st.selectbox("Select the label column", data.columns)
        prepared = data[[text_col, label_col]].dropna().copy()
        prepared.columns = ["text", "label_raw"]
        prepared["text"] = prepared["text"].astype(str).str.strip()
        prepared["label"] = prepared["label_raw"].map(normalize_label)
        prepared = prepared[(prepared.text.str.len() > 0) & prepared.label.notna()]
        if prepared.label.nunique() < 2:
            st.error("The selected label column must contain examples of both Fake and Genuine reviews.")
        elif prepared.label.value_counts().min() < 2:
            st.error("Please provide at least two examples for each label.")
        else:
            if st.button("Train model", type="primary"):
                model = train_model(tuple(prepared.text), tuple(prepared.label))
                st.session_state["review_model"] = model
                st.session_state["trained_rows"] = len(prepared)
                st.success(f"Model trained on {len(prepared)} labeled reviews.")
            if "review_model" in st.session_state:
                model = st.session_state["review_model"]
                st.success(f"Model ready ({st.session_state['trained_rows']} training reviews).")
    except Exception as exc:
        st.error(f"Could not read that CSV: {exc}")
# Restore trained model after Streamlit rerun
if "review_model" in st.session_state:
    model = st.session_state["review_model"]

# Check a review
st.subheader("Check a review")

review = st.text_area(
    "Paste a product review",
    height=150,
    placeholder="For example: The headphones stopped working after two days..."
)

if st.button("Analyze review", type="primary"):

    if not review.strip():
        st.warning("Enter a review first.")

    elif model is None:
        st.warning("Upload a labeled CSV and train the model first.")

    else:
        prediction = model.predict([review])[0]

        probabilities = model.predict_proba([review])[0]
        classes = list(model.classes_)
        confidence = float(
            probabilities[classes.index(prediction)]
        )

        if prediction == "Fake":
            st.error(
                f"Likely Fake Review - Confidence: {confidence:.0%}"
            )
        else:
            st.success(
                f"Likely Genuine Review - Confidence: {confidence:.0%}"
            )