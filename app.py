import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv('https://raw.githubusercontent.com/SiddharthReddy264/datasets/main/news.csv')

# Preprocess
x = df['text']
y = df['label']
tfidf = TfidfVectorizer(stop_words='english', max_df=0.7)
x_tfidf = tfidf.fit_transform(x)
x_train, x_test, y_train, y_test = train_test_split(x_tfidf, y, test_size=0.2)

# Train model
model = PassiveAggressiveClassifier()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

# UI
st.title("📰 Fake News Detector")
news = st.text_area("Enter News Article Text:")
if st.button("Check"):
    if news.strip() == "":
        st.warning("Please enter some text.")
    else:
        tfidf_news = tfidf.transform([news])
        prediction = model.predict(tfidf_news)
        st.success(f"This news is **{prediction[0]}**")
        
st.write("Model Accuracy:", accuracy_score(y_test, y_pred))
