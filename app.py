import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords

# Download stopwords if not already
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load model & vectorizer
model = joblib.load("best_sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Function to clean text
def clean_text(text):
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = text.lower()
    tokens = [w for w in text.split() if w not in stop_words]
    return " ".join(tokens)

# Streamlit UI
st.title("IMDB Movie Review Sentiment Analysis")
st.write("Enter a movie review below to predict its sentiment.")

review = st.text_area("Movie Review", "")
if st.button("Predict"):
    if review:
        cleaned_review = clean_text(review)
        vec = vectorizer.transform([cleaned_review])
        prediction = model.predict(vec)
        sentiment = "Positive" if prediction[0] == 1 else "Negative"
        st.success(f"Predicted Sentiment: {sentiment}")
    else:
        st.warning("Please enter a review!")
