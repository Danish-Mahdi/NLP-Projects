import os
import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer

nltk.download('stopwords')  

# Load the pre-trained model and vectorizer
# model = joblib.load('Restaurant_review_model.joblib')

# check
# Get the current script directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the absolute path to the model
model_path = os.path.join(script_dir, 'Restaurant_review_model.joblib')
# model = joblib.load(script_dir)
vectorizer_path = os.path.join(script_dir, 'count_v_res.joblib')
vectorizer = joblib.load(vectorizer_path)
# check
# vectorizer = joblib.load('count_v_res.joblib')

def preprocess_text(text):
    custom_stopwords = {'don', "don't", 'ain', 'aren', "aren't", 'couldn', "couldn't",
                        'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't",
                        'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't",
                        'needn', "needn't", 'shan', "shan't", 'no', 'nor', 'not', 'shouldn', "shouldn't",
                        'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"}
    ps = PorterStemmer()
    stop_words = set(stopwords.words("english")) - custom_stopwords

    review = re.sub('[^a-zA-Z]', ' ', text)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if word not in stop_words]
    review = " ".join(review)

    return review

st.title("Restaurant Review Classification App")

user_input = st.text_area("Enter your restaurant review:")

if st.button("Classify"):
    if user_input:
        processed_input = preprocess_text(user_input)
        processed_input_vectorized = vectorizer.transform([processed_input])
        prediction = model.predict(processed_input_vectorized)[0]
        sentiment = "Positive" if prediction == 1 else "Negative"
        st.success(f"Predicted Sentiment: {sentiment}")
    else:
        st.warning("Please enter a review before clicking 'Classify'.")
