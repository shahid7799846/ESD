
import streamlit as st
import joblib
import re

# Load trained model and vectorizer
model = joblib.load("spam_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

# Clean email text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Configure application
st.set_page_config(
    page_title="Email Spam Detection",
    page_icon="📧"
)

st.title("📧 Email Spam Detection System")
st.write("Check whether your email is Spam or Normal.")

email = st.text_area("Enter your email message:")

if st.button("Check Email"):
    if email.strip() == "":
        st.warning("Please enter an email.")
    else:
        cleaned_email = clean_text(email)
        email_features = tfidf.transform([cleaned_email])
        prediction = model.predict(email_features)[0]

        if prediction == "spam":
            st.error("This email is SPAM!")
        else:
            st.success("This email is NORMAL (HAM).")

