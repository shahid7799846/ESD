
import streamlit as st
import joblib
import re
import pandas as pd
from datetime import datetime

# =========================================
# 1. PAGE CONFIGURATION
# =========================================

st.set_page_config(
    page_title="MailGuard AI",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================
# 2. LOAD MODEL AND VECTORIZER
# =========================================

@st.cache_resource
def load_model():
    model = joblib.load("spam_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return model, vectorizer

try:
    model, tfidf = load_model()
except Exception as e:
    st.error("Unable to load model files.")
    st.code(str(e))
    st.stop()

# =========================================
# 3. SESSION STATE
# =========================================

if "history" not in st.session_state:
    st.session_state.history = []

if "result" not in st.session_state:
    st.session_state.result = None

# =========================================
# 4. TEXT CLEANING
# =========================================

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# =========================================
# 5. CUSTOM LIGHT DASHBOARD DESIGN
# =========================================

st.markdown("""
<style>

/* Main background */

.stApp {
    background: #F5F8FE;
    color: #1E293B;
}

/* Main content */

.block-container {
    max-width: 1250px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Sidebar */

[data-testid="stSidebar"] {
    background: #FFFFFF;
    border-right: 1px solid #E6ECF5;
}

/* Hero section */

.hero {
    background: linear-gradient(120deg, #FFFFFF, #EAF2FF);
    border: 1px solid #DCE7F7;
    border-radius: 20px;
    padding: 32px;
    margin-bottom: 25px;
}

.hero h1 {
    color: #173B75;
    font-size: 34px;
    font-weight: 750;
    margin-bottom: 10px;
}

.hero p {
    color: #64748B;
    font-size: 15px;
}

/* Cards */

.card {
    background: #FFFFFF;
    border: 1px solid #E5EAF2;
    border-radius: 16px;
    padding: 22px;
    box-shadow: 0 3px 12px rgba(30, 64, 120, 0.04);
}

.card-label {
    color: #64748B;
    font-size: 13px;
    font-weight: 500;
}

.card-value {
    color: #173B75;
    font-size: 29px;
    font-weight: 750;
    margin-top: 8px;
}

/* Section headings */

.section-heading {
    color: #173B75;
    font-size: 22px;
    font-weight: 700;
    margin-top: 15px;
    margin-bottom: 12px;
}

/* Email panel */

.email-panel {
    background: #FFFFFF;
    border: 1px solid #E1E8F2;
    border-radius: 18px;
    padding: 22px;
}

/* Result cards */

.spam-result {
    background: #FFF1F2;
    border: 1px solid #FECDD3;
    border-radius: 16px;
    padding: 22px;
    color: #9F1239;
}

.ham-result {
    background: #ECFDF5;
    border: 1px solid #BBF7D0;
    border-radius: 16px;
    padding: 22px;
    color: #166534;
}

/* Buttons */

.stButton > button {
    background: #2563EB;
    color: #FFFFFF;
    border: none;
    border-radius: 10px;
    min-height: 48px;
    font-size: 15px;
    font-weight: 650;
    transition: 0.2s;
}

.stButton > button:hover {
    background: #1D4ED8;
    color: #FFFFFF;
    border: none;
}

/* Text area */

.stTextArea textarea {
    background: #FFFFFF;
    border: 1px solid #DCE4F0;
    border-radius: 12px;
    color: #1E293B;
    font-size: 15px;
}

/* Footer */

.footer {
    text-align: center;
    color: #94A3B8;
    font-size: 12px;
    padding: 25px 0;
}

</style>
""", unsafe_allow_html=True)

# =========================================
# 6. SIDEBAR NAVIGATION
# =========================================

with st.sidebar:

    st.markdown(
        "<h2 style='color:#2563EB;'>📧 MailGuard AI</h2>",
        unsafe_allow_html=True
    )

    st.caption("Intelligent Email Security")

    st.divider()

    page = st.radio(
        "MAIN MENU",
        [
            "🏠 Dashboard",
            "📊 History",
            "🤖 About"
        ],
        label_visibility="visible"
    )

    st.divider()

    st.markdown("### SYSTEM STATUS")

    st.success("Machine Learning Model Active")

    st.markdown("**Algorithm**")
    st.write("Naive Bayes")

    st.markdown("**Feature Extraction**")
    st.write("TF-IDF")

    st.markdown("**Classification**")
    st.write("Spam / Ham")

    st.divider()

    st.caption("Email Spam Detection Project")

# =========================================
# 7. DASHBOARD
# =========================================

if page == "🏠 Dashboard":

    # Hero section

    st.markdown("""
    <div class="hero">
        <h1>📧 Email Spam Detection</h1>
        <p>
        Welcome to MailGuard AI. Analyze email messages
        using machine learning and identify Spam or
        Normal emails instantly.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Dashboard statistics

    total = len(st.session_state.history)

    spam_total = sum(
        1 for x in st.session_state.history
        if x["Prediction"] == "SPAM"
    )

    ham_total = total - spam_total

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(f"""
        <div class="card">
            <div class="card-label">TOTAL EMAILS ANALYZED</div>
            <div class="card-value">{total}</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="card">
            <div class="card-label">SPAM DETECTED</div>
            <div class="card-value" style="color:#E11D48;">
                {spam_total}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="card">
            <div class="card-label">NORMAL EMAILS</div>
            <div class="card-value" style="color:#16A34A;">
                {ham_total}
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    # Email input section

    st.markdown("""
    <div class="section-heading">
        Analyze Your Email
    </div>
    """, unsafe_allow_html=True)

    st.caption(
        "Paste your email message below to identify whether it is Spam or Normal."
    )

    email = st.text_area(
        "Email Message",
        placeholder="Dear user, enter or paste your email content here...",
        height=190,
        label_visibility="collapsed"
    )

    st.write("")

    analyze = st.button(
        "🔍  Analyze Email",
        use_container_width=True
    )

    # =====================================
    # 8. EMAIL PREDICTION
    # =====================================

    if analyze:

        if not email.strip():

            st.warning("Please enter an email message.")

        else:

            with st.spinner("Analyzing your email..."):

                cleaned_email = clean_text(email)

                features = tfidf.transform([cleaned_email])

                prediction = model.predict(features)[0]

                probabilities = model.predict_proba(features)[0]

                classes = model.classes_

                probability_dict = {
                    str(c).lower(): float(p)
                    for c, p in zip(classes, probabilities)
                }

                prediction = str(prediction).lower()

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
                    ),
                    "HAM Probability": ham_probability,
                    "SPAM Probability": spam_probability
                }

                st.session_state.result = result

                st.session_state.history.append({
                    "Email": email[:100],
                    "Prediction": prediction.upper(),
                    "Confidence": round(confidence, 2),
                    "Time": result["Time"]
                })

    # =====================================
    # 9. DISPLAY RESULT
    # =====================================

    if st.session_state.result is not None:

        result = st.session_state.result

        st.divider()

        st.markdown("""
        <div class="section-heading">
            Analysis Results
        </div>
        """, unsafe_allow_html=True)

        if result["Prediction"] == "SPAM":

            st.markdown("""
            <div class="spam-result">
                <h2>🚨 SPAM EMAIL DETECTED</h2>
                <p>
                The machine learning model classified
                this message as Spam.
                </p>
            </div>
            """, unsafe_allow_html=True)

        else:

            st.markdown("""
            <div class="ham-result">
                <h2>✅ NORMAL EMAIL DETECTED</h2>
                <p>
                The machine learning model classified
                this message as Normal (Ham).
                </p>
            </div>
            """, unsafe_allow_html=True)

        st.write("")

        # Confidence card

        r1, r2 = st.columns(2)

        with r1:

            st.markdown(f"""
            <div class="card">
                <div class="card-label">
                    PREDICTION CONFIDENCE
                </div>
                <div class="card-value">
                    {result["Confidence"]:.2f}%
                </div>
            </div>
            """, unsafe_allow_html=True)

        with r2:

            st.markdown(f"""
            <div class="card">
                <div class="card-label">
                    PREDICTED CATEGORY
                </div>
                <div class="card-value">
                    {result["Prediction"]}
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.write("")

        # Probability visualization

        st.markdown("""
        <div class="section-heading">
            Probability Distribution
        </div>
        """, unsafe_allow_html=True)

        p1, p2 = st.columns(2)

        with p1:

            st.markdown("**🟢 Normal Email (HAM)**")

            st.progress(
                result["HAM Probability"] / 100
            )

            st.caption(
                f'{result["HAM Probability"]:.2f}%'
            )

        with p2:

            st.markdown("**🔴 Spam Email**")

            st.progress(
                result["SPAM Probability"] / 100
            )

            st.caption(
                f'{result["SPAM Probability"]:.2f}%'
            )

        st.caption(
            "Probabilities are estimates from the trained model "
            "and do not guarantee prediction correctness."
        )

# =========================================
# 10. HISTORY PAGE
# =========================================

elif page == "📊 History":

    st.markdown("""
    <div class="hero">
        <h1>📊 Prediction History</h1>
        <p>Review emails analyzed during this session.</p>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.history:

        history_df = pd.DataFrame(
            st.session_state.history
        )

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )

        csv_file = history_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "⬇️ Download History",
            data=csv_file,
            file_name="email_history.csv",
            mime="text/csv"
        )

        if st.button("🗑️ Clear History"):

            st.session_state.history = []
            st.session_state.result = None

            st.rerun()

    else:

        st.info("No emails analyzed yet.")

# =========================================
# 11. ABOUT PAGE
# =========================================

elif page == "🤖 About":

    st.markdown("""
    <div class="hero">
        <h1>🤖 About MailGuard AI</h1>
        <p>Machine Learning Based Email Spam Detection System</p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Project Overview")

    st.write("""
    MailGuard AI is a machine learning application
    that classifies email messages into two categories:
    Spam and Normal (Ham).
    """)

    st.subheader("Technologies Used")

    st.markdown("""
    - Python
    - Pandas
    - Scikit-learn
    - TF-IDF Vectorization
    - Multinomial Naive Bayes
    - Streamlit
    - GitHub
    """)

    st.subheader("Working Methodology")

    st.markdown("""
    1. Email message is entered by the user.
    2. Text preprocessing is performed.
    3. TF-IDF converts text into numerical features.
    4. Naive Bayes predicts the email category.
    5. The result and estimated probabilities are displayed.
    """)

    st.info(
        "This project is an educational demonstration. "
        "It may occasionally misclassify messages."
    )

# =========================================
# 12. FOOTER
# =========================================

st.markdown("""
<div class="footer">
    <b>MAILGUARD AI</b><br>
    Email Spam Detection System<br>
    Developed using Python and Machine Learning
</div>
""", unsafe_allow_html=True)
