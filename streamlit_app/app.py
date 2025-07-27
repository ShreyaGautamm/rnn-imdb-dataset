# app.py

# Step 1: Import Libraries and Load the Model
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

# Load the IMDB dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Load the pre-trained model
model = load_model('models/simple_rnn_imdb.h5')  # Update path if needed

# Step 2: Helper Functions
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

# Step 3: Streamlit UI
st.set_page_config(page_title="🎬 IMDB Sentiment Classifier", page_icon="🎭")

# Custom styling
st.markdown(
    """
    <style>
        .reportview-container {
            padding: 2rem 2rem 2rem 2rem;
            font-family: 'Segoe UI', sans-serif;
        }
        .stTextArea label {
            font-size: 18px;
        }
        .big-font {
            font-size:28px !important;
            color: #FF6347;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<h1 style='text-align: center;'>🎬 IMDB Movie Review Sentiment Analysis 🎭</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Enter a review and let the model classify it as Positive or Negative using a trained SimpleRNN model.</p>", unsafe_allow_html=True)

# User Input
user_input = st.text_area("📝 Enter your movie review below:")

# Prediction
if st.button("🔍 Classify Review"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a review before submitting.")
    else:
        preprocessed_input = preprocess_text(user_input)
        prediction = model.predict(preprocessed_input)[0][0]

        # Set color and emoji based on prediction
        if prediction > 0.5:
            sentiment = "🟢 Positive"
            color = "green"
        else:
            sentiment = "🔴 Negative"
            color = "red"

        st.markdown(f"<h3 style='color:{color}'>Prediction: {sentiment}</h3>", unsafe_allow_html=True)
        st.markdown(f"**Confidence Score:** `{prediction:.4f}`")
else:
    st.info("👈 Enter your review and click **Classify Review** to begin.")
