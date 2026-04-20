import streamlit as st
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# -------------------------------
# Sample Twitter Data (Training)
# -------------------------------
tweets = [
    "Government announces new policy",
    "Official news released today",
    "Click here to win money now",
    "Fake celebrity scandal click now",
    "Ministry confirms update",
    "You won lottery claim prize"
]

labels = [1, 1, 0, 0, 1, 0]

# -------------------------------
# Preprocessing
# -------------------------------
def clean_text(text):
    text = text.lower()
    text = "".join([c for c in text if c not in string.punctuation])
    return text

cleaned = [clean_text(t) for t in tweets]

# -------------------------------
# Feature Extraction (TF-IDF)
# -------------------------------
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(cleaned)

# -------------------------------
# Model Training (Logistic Regression)
# -------------------------------
model = LogisticRegression()
model.fit(X, labels)

# -------------------------------
# UI (User Input Module)
# -------------------------------
st.title("🐦 Fake News Detection on Twitter")
st.write("Enter a tweet to analyze whether it is **REAL** or **FAKE**")

user_input = st.text_area("Enter Tweet")

# -------------------------------
# Prediction + Output Module
# -------------------------------
if st.button("Analyze Tweet"):
    if user_input.strip() != "":
        cleaned_input = clean_text(user_input)
        vector = vectorizer.transform([cleaned_input])

        prediction = model.predict(vector)
        prob = model.predict_proba(vector)
        confidence = max(prob[0]) * 100

        # Output Result
        if prediction[0] == 1:
            st.success(f"✅ Real Tweet (Confidence: {confidence:.2f}%)")
        else:
            st.error(f"❌ Fake Tweet (Confidence: {confidence:.2f}%)")

        # -------------------------------
        # EXTRA FEATURES (REVISED KEYWORDS)
        # -------------------------------
        st.subheader("🔍 Analysis")

        # Tweet Length
        st.write(f"Tweet Length: {len(user_input.split())} words")

        # Expanded Red Flag Keywords (Approx. 75 keywords)
        fake_keywords = [
            "click", "win", "free", "lottery", "shocking", "freebies", "accidents",
            "money", "cash", "prize", "crypto", "bitcoin", "giveaway", "bonus",
            "urgent", "immediately", "secret", "hidden", "truth", "exposed",
            "scandal", "leaked", "miracle", "cure", "weight loss", "billionaire",
            "investment", "guaranteed", "risk-free", "double", "triple", "easy",
            "simple", "amazing", "unbelievable", "wow", "omg", "alert", "warning",
            "scam", "fraud", "hack", "phishing", "login", "password", "verify",
            "account", "suspended", "exclusive", "offer", "discount", "coupon",
            "voucher", "claim", "register", "signup", "subscribe", "clickhere",
            "linkinbio", "followback", "followers", "retweets", "share", "viral",
            "breaking", "insider", "classified", "dying", "dead", "unreal", 
            "congratulations", "inheritance", "payment", "bank", "transfer"
        ]

        # Detection logic
        detected_flags = [word for word in fake_keywords if word in cleaned_input.split()]

        if detected_flags:
            st.warning(f"⚠️ Suspicious Words Detected: {', '.join(detected_flags)}")
        else:
            st.info("No suspicious keywords detected")

        # Simple reasoning
        st.subheader("🧠 Reasoning")
        if prediction[0] == 0:
            st.write("This tweet contains patterns commonly associated with clickbait, scams, or sensationalism.")
        else:
            st.write("This tweet follows a formal and informational structure typical of verified news sources.")

    else:
        st.warning("Please enter a tweet")
