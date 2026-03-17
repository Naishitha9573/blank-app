import streamlit as st
import pickle

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# App Title
st.set_page_config(page_title="Fake News Detector", page_icon="📰")
st.title("📰 Fake News Detection App")

st.write("Enter a news article or headline to check whether it is **Real or Fake**.")

# Input text
news_input = st.text_area("Enter News Text:")

# Prediction function
def predict_news(text):
    vector = vectorizer.transform([text])
    prediction = model.predict(vector)
    return prediction[0]

# Button
if st.button("Check News"):
    if news_input.strip() == "":
        st.warning("⚠️ Please enter some text!")
    else:
        result = predict_news(news_input)

        if result == 1:
            st.success("✅ This News is REAL")
        else:
            st.error("❌ This News is FAKE")
