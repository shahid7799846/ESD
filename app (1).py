
import streamlit as st
import joblib
import re
import pandas as pd
from datetime import datetime

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="MailGuard AI | Email Spam Detection",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. LOAD TRAINED MODEL AND VECTORIZER
# ==========================================

@st.cache_resource
def load_models():
    model = joblib.load("spam_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return model, vectorizer

try:
    model, tfidf = load_models()
except Exception as e:
    st.error("Unable to load the trained model files.")
    st.error(str(e))
    st.stop()

# ==========================================
# 3. SESSION STATE
# ==========================================

if "history" not in st.session_state:
    st.session_state.history = []

if "last_result" not in st.session_state:
    st.session_state.last_result = None

# ==========================================
# 4. TEXT CLEANING FUNCTION
# ==========================================

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ==========================================
# 5. CUSTOM CSS DESIGN
# ==========================================

st.markdown("""
<style>

.stApp {
    background-color: #F5F8FC;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1250px;
}

[data-testid="stSidebar"] {
    background-color: #FFFFFF;
    border-right: 1px solid #E5EAF2;
}

.hero {
    background: linear-gradient(120deg, #FFFFFF, #EAF2FF);
    padding: 30px;
    border-radius: 18px;
    border: 1px solid #DCE7F7;
    margin-bottom: 25px;
}

.hero h1 {
    color: #173B75;
    font-size: 34px;
    margin-bottom: 8px;
}

.hero p {
    color: #64748B;
    font-size: 15px;
}

.metric-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 14px;
    padding: 20px;
    min-height: 115px;
}

.metric-label {
    color: #64748B;
    font-size: 13px;
}

.metric-value {
    color: #1E3A8A;
    font-size: 27px;
    font-weight: 700;
    margin-top: 8px;
}

.result-spam {
    background: #FFF1F2;
    border: 1px solid #FECDD3;
    padding: 20px;
    border-radius: 14px;
    color: #9F1239;
}

.result-ham {
    background: #F0FDF4;
    border: 1px solid #BBF7D0;
    padding: 20px;
    border-radius: 14px;
    color: #166534;
}

.section-title {
    color: #173B75;
    font-size: 21px;
    font-weight: 700;
}

.footer {
    text-align: center;
    color: #94A3B8;
    font-size: 12px;
    padding: 20px;
}

.stButton > button {
    background-color: #2563EB;
    color: white;
    border-radius: 9px;
    border: none;
    font-weight: 600;
    min-height: 45px;
}

.stButton > button:hover {
    background-color: #1D4ED8;
    color: white;
    border: none;
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# 6. SIDEBAR
# ==========================================

with st.sidebar:

    st.markdown("## 📧 MailGuard AI")

    st.caption("Email Security Intelligence")

    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "🏠 Dashboard",
            "📊 Prediction History",
            "🤖 About Model"
        ]
    )

    st.divider()

    st.markdown("### Model Information")

    st.success("Model Loaded")

    st.write("**Algorithm:** Naive Bayes")
    st.write("**Features:** TF-IDF")
    st.write("**Task:** Binary Classification")

    st.divider()

    st.caption("Developed as an AIML project")

# ==========================================
# 7. DASHBOARD PAGE
# ==========================================

if page == "🏠 Dashboard":

    st.markdown("""
    <div class="hero">
        <h1>📧 Email Spam Detection</h1>
        <p>
        Protect your inbox with Machine Learning.
        Analyze email content and classify it as
        Spam or Normal (Ham).
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Summary metrics
    total = len(st.session_state.history)

    spam_count = sum(
        1 for item in st.session_state.history
        if item["Prediction"] == "SPAM"
    )

    ham_count = total - spam_count

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Emails Analyzed</div>
            <div class="metric-value">{total}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Spam Emails Detected</div>
            <div class="metric-value">{spam_count}</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Normal Emails</div>
            <div class="metric-value">{ham_count}</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.markdown(
        '<div class="section-title">Analyze Your Email</div>',
        unsafe_allow_html=True
    )

    st.caption(
        "Paste the email content below to check whether it is Spam or Normal."
    )

    email = st.text_area(
        "Email Message",
        height=190,
        placeholder="Enter or paste your email message here..."
    )

    analyze = st.button(
        "🔍 Analyze Email",
        use_container_width=True
    )

    if analyze:

        if not email.strip():
            st.warning("Please enter an email message.")

        else:

            with st.spinner("Analyzing email..."):

                cleaned_email = clean_text(email)

                email_features = tfidf.transform(
                    [cleaned_email]
                )

                prediction = model.predict(
                    email_features
                )[0]

                probabilities = model.predict_proba(
                    email_features
                )[0]

                classes = model.classes_

                probability_dict = dict(
                    zip(classes, probabilities)
                )

                confidence = (
                    probability_dict[prediction] * 100
                )

                ham_probability = (
                    probability_dict.get("ham", 0) * 100
                )

                spam_probability = (
                    probability_dict.get("spam", 0) * 100
                )

                result = {
                    "Email": email[:100],
                    "Prediction": prediction.upper(),
                    "Confidence": round(confidence, 2),
                    "Time": datetime.now().strftime(
                        "%d-%m-%Y %H:%M:%S"
                    )
                }

                st.session_state.history.append(result)

                st.session_state.last_result = {
                    **result,
                    "HAM Probability": ham_probability,
                    "SPAM Probability": spam_probability
                }

    # Show latest result
    if st.session_state.last_result is not None:

        result = st.session_state.last_result

        st.divider()

        st.markdown(
            '<div class="section-title">Analysis Results</div>',
            unsafe_allow_html=True
        )

        if result["Prediction"] == "SPAM":

            st.markdown("""
            <div class="result-spam">
                <h2>🚨 SPAM EMAIL DETECTED</h2>
                <p>This email was classified as Spam by the model.</p>
            </div>
            """, unsafe_allow_html=True)

        else:

            st.markdown("""
            <div class="result-ham">
                <h2>✅ NORMAL EMAIL (HAM)</h2>
                <p>This email was classified as Normal by the model.</p>
            </div>
            """, unsafe_allow_html=True)

        st.write("")

        c1, c2 = st.columns(2)

        with c1:
            st.metric(
                "Prediction Confidence",
                f'{result["Confidence"]:.2f}%'
            )

        with c2:
            st.metric(
                "Predicted Class",
                result["Prediction"]
            )

        st.markdown("### Class Probability Distribution")

        p1, p2 = st.columns(2)

        with p1:
            st.write("**HAM**")
            st.progress(
                min(1.0, max(0.0, result["HAM Probability"] / 100))
            )
            st.caption(
                f'{result["HAM Probability"]:.2f}%'
            )

        with p2:
            st.write("**SPAM**")
            st.progress(
                min(1.0, max(0.0, result["SPAM Probability"] / 100))
            )
            st.caption(
                f'{result["SPAM Probability"]:.2f}%'
            )

        st.caption(
            "Confidence represents the model's estimated probability, "
            "not a guarantee of correctness."
        )

# ==========================================
# 8. PREDICTION HISTORY PAGE
# ==========================================

elif page == "📊 Prediction History":

    st.markdown(
        '<div class="section-title">📊 Prediction History</div>',
        unsafe_allow_html=True
    )

    st.caption(
        "Previously analyzed emails in this active session."
    )

    if st.session_state.history:

        history_df = pd.DataFrame(
            st.session_state.history
        )

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )

        st.write(
            f"Total analyzed emails: {len(history_df)}"
        )

        csv_data = history_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "⬇️ Download History as CSV",
            data=csv_data,
            file_name="email_prediction_history.csv",
            mime="text/csv"
        )

        if st.button("🗑️ Clear History"):

            st.session_state.history = []
            st.session_state.last_result = None

            st.rerun()

    else:

        st.info(
            "No prediction history yet. "
            "Analyze an email from the Dashboard."
        )

# ==========================================
# 9. ABOUT MODEL PAGE
# ==========================================

elif page == "🤖 About Model":

    st.markdown(
        '<div class="section-title">🤖 About the ML Model</div>',
        unsafe_allow_html=True
    )

    st.write(
        "This application uses Machine Learning "
        "to classify email messages into two categories."
    )

    st.subheader("Technology Used")

    st.markdown("""
    - **Python:** Programming language
    - **Pandas:** Data handling
    - **Scikit-learn:** Machine Learning
    - **TF-IDF:** Text feature extraction
    - **Multinomial Naive Bayes:** Classification algorithm
    - **Streamlit:** Web application framework
    """)

    st.subheader("How the System Works")

    st.markdown("""
    1. User enters an email message.
    2. The text is cleaned.
    3. TF-IDF converts text into numerical features.
    4. The trained Naive Bayes model predicts the category.
    5. The application displays the result and estimated probability.
    """)

    st.subheader("Classification Categories")

    st.success("HAM: Normal or legitimate email.")

    st.error("SPAM: Email classified as unwanted or suspicious.")

    st.info(
        "This system is a machine learning demonstration. "
        "Its predictions may sometimes be incorrect."
    )

# ==========================================
# 10. FOOTER
# ==========================================

st.markdown("""
<div class="footer">
    MailGuard AI | Email Spam Detection System<br>
    Built using Python, TF-IDF, Naive Bayes and Streamlit
</div>
""", unsafe_allow_html=True)
