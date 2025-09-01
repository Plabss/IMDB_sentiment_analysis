# IMDB Movie Sentiment Analysis

This project predicts whether an IMDB movie review is **positive** or **negative** using machine learning.  
It demonstrates the full process of working with text data: cleaning and preprocessing text, turning it into features, training and evaluating models, visualizing results, and saving the best model for a live app.

---

## Live Demo

[![Try it Live](https://img.shields.io/badge/Streamlit-Live-blue?style=for-the-badge&logo=streamlit)](https://imdbsentimentanalysis-plabon.streamlit.app/)

---

## Dataset
- Source: [IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)  
- Contains 50,000 reviews labeled as `positive` or `negative`.

---

## Project Structure
- `IMDB_sentiment_analysis.ipynb` - Complete code pipeline for training, evaluating, and visualizing models.  
- `best_sentiment_model.pkl` - Saved best model.  
- `tfidf_vectorizer.pkl` - Saved TF-IDF vectorizer.  
- `README.md` - Project overview.
- `requirements.txt` – Dependencies 
---

## Key Steps
1. **Data Loading & EDA**  
   Load dataset, check class distribution, visualize sentiment counts.

2. **Text Preprocessing**  
   Clean text: remove HTML tags, non-alphabetic characters, lowercase, remove stopwords.

3. **Feature Extraction**  
   Convert text to numeric features using **TF-IDF** vectorization.

4. **Model Training & Evaluation**  
   Models used: Logistic Regression, Naive Bayes, Random Forest, SVM, XGBoost  
   Metrics: Accuracy, Precision, Recall, F1  
   Confusion matrix visualizations for all models

5. **Visualization**  
   Interactive plots to compare metrics across models

6. **Save Best Model**  
   Model with highest **F1 score** is saved along with TF-IDF vectorizer for deployment

---

## Technologies Used
- **Python** – Programming language  
- **Pandas, NumPy** – Data manipulation  
- **NLTK** – Text preprocessing and stopwords  
- **Scikit-learn** – Machine learning models and TF-IDF vectorization  
- **Joblib** – Model and vectorizer serialization  
- **Streamlit** – Web app deployment  
- **Matplotlib, Seaborn** – Visualizations
