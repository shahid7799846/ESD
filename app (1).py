
import streamlit as st
import joblib
import re
import pandas as pd

# Load model and vectorizer
model = joblib.load("spam_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")


# Initialize prediction history
if "history" not in st.session_state:
    st.session_state.history = []


# Clean email text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# Page configuration
st.set_page_config(
    page_title="Email Spam Detector",
    page_icon="📧",
    layout="centered"
)

# Heading
st.title("📧 Email Spam Detector")
st.write("AI-powered Email Classification System")

st.divider()


# Email input
st.subheader("Enter Your Email")

email = st.text_area(
    "Paste your email:",
    height=150
)


# Analyze button
if st.button("🔍 Analyze Email", use_container_width=True):

    if email.strip() == "":
        st.warning("Please enter an email.")

    else:
        cleaned_email = clean_text(email)

        email_features = tfidf.transform([cleaned_email])

        prediction = model.predict(email_features)[0]

        probabilities = model.predict_proba(email_features)[0]

        classes = model.classes_

        confidence = (
            probabilities[list(classes).index(prediction)] * 100
        )

        # Save prediction in history
        st.session_state.history.append({
            "Email": email[:100],
            "Prediction": prediction.upper(),
            "Confidence": round(confidence, 2)
        })

        # Display prediction
        st.subheader("Prediction Result")

        if prediction == "spam":
            st.error("🚨 SPAM EMAIL DETECTED")
        else:
            st.success("✅ NORMAL EMAIL (HAM)")

        st.metric(
            "Prediction Confidence",
            f"{confidence:.2f}%"
        )


# Prediction history
st.divider()

st.subheader("📊 Prediction History")

if len(st.session_state.history) > 0:

    history_df = pd.DataFrame(st.session_state.history)

    st.dataframe(
        history_df,
        use_container_width=True
    )

    st.write(
        "Total Emails Analyzed:",
        len(st.session_state.history)
    )

    # Clear history button
    if st.button("🗑️ Clear History"):
        st.session_state.history = []
        st.rerun()

else:
    st.info("No prediction history available yet.")


# Footer
st.divider()

st.caption("Email Spam Detection | AIML Project")
