
import streamlit as st
import joblib
import re

# -----------------------------
# 1. Load trained ML model
# -----------------------------

model = joblib.load("spam_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")


# -----------------------------
# 2. Text cleaning function
# -----------------------------

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# -----------------------------
# 3. Streamlit page
# -----------------------------

st.set_page_config(
    page_title="Email Spam Detection",
    page_icon="📧"
)

st.title("📧 Email Spam Detection System")

st.write(
    "Check whether an email is Spam or Normal."
)

# -----------------------------
# 4. Email input
# -----------------------------

email = st.text_area("Enter your email message:")


# -----------------------------
# 5. Prediction button
# -----------------------------

if st.button("Check Email"):

    if email.strip() == "":
        st.warning("Please enter an email.")

    else:
        # Clean email
        cleaned_email = clean_text(email)

        # Convert text into TF-IDF features
        email_features = tfidf.transform([cleaned_email])

        # Predict category
        prediction = model.predict(email_features)[0]

        # Calculate probabilities
        probabilities = model.predict_proba(email_features)[0]

        # Get class names
        classes = model.classes_

        # Calculate confidence
        confidence = (
            probabilities[list(classes).index(prediction)] * 100
        )

        # Display prediction
        if prediction == "spam":
            st.error("🚨 This email is SPAM!")
        else:
            st.success("✅ This email is NORMAL (HAM).")

        # Display confidence
        st.subheader("Prediction Confidence")

        st.metric(
            label="Model Confidence",
            value=f"{confidence:.2f}%"
        )

        # Display probabilities
        st.subheader("Class Probabilities")

        for class_name, probability in zip(classes, probabilities):
            st.write(
                f"{class_name.upper()}: {probability * 100:.2f}%"
            )
