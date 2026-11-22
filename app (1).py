
import streamlit as st
import joblib
import re

# Load trained model and vectorizer
model = joblib.load("spam_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")


# Text cleaning function
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

# Custom CSS
st.markdown("""
<style>
.main {
    background-color: #f5f7fb;
}
.title {
    text-align: center;
    color: #2563eb;
    font-size: 35px;
    font-weight: bold;
}
.subtitle {
    text-align: center;
    color: gray;
    font-size: 16px;
}
</style>
""", unsafe_allow_html=True)


# Application heading
st.markdown(
    '<div class="title">📧 Email Spam Detector</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI-powered Email Classification System</div>',
    unsafe_allow_html=True
)

st.divider()


# Email input section
st.subheader("Enter Your Email")

email = st.text_area(
    "Paste your email message below:",
    height=180,
    placeholder="Enter email content here..."
)


# Prediction
if st.button("🔍 Analyze Email", use_container_width=True):

    if email.strip() == "":
        st.warning("Please enter an email message.")

    else:
        cleaned_email = clean_text(email)

        email_features = tfidf.transform([cleaned_email])

        prediction = model.predict(email_features)[0]

        probabilities = model.predict_proba(email_features)[0]

        classes = model.classes_

        confidence = (
            probabilities[list(classes).index(prediction)] * 100
        )

        # Display prediction
        st.divider()
        st.subheader("Analysis Results")

        if prediction == "spam":
            st.error("🚨 SPAM EMAIL DETECTED")
        else:
            st.success("✅ NORMAL EMAIL (HAM)")

        # Confidence section
        st.write("Prediction Confidence")

        st.progress(int(confidence))

        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )

        # Probability section
        st.subheader("Class Probabilities")

        col1, col2 = st.columns(2)

        probability_dict = dict(zip(classes, probabilities))

        ham_probability = probability_dict.get("ham", 0) * 100
        spam_probability = probability_dict.get("spam", 0) * 100

        col1.metric(
            "HAM",
            f"{ham_probability:.2f}%"
        )

        col2.metric(
            "SPAM",
            f"{spam_probability:.2f}%"
        )

        st.divider()

        st.caption(
            "Powered by Machine Learning | Naive Bayes + TF-IDF"
        )

# Footer
st.divider()

st.caption("Email Spam Detection Project | AIML")
