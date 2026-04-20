import streamlit as st
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# -------------------------------
# 1. DATA PREPARATION (FIXED)
# -------------------------------

# Create a clean list of Real Tweets
real_tweets = [
    "Government announces new policy",
    "Official news released today",
    "Ministry confirms update",
    "The city council approved the new budget",
    "Weather update: Heavy rain expected tomorrow",
    "Health officials report decline in flu cases"
]

# Create your list of Fake Tweets
fake_tweets = [
    "Click here to win $5,000 cash instantly!",
    "CONGRATULATIONS! You have been selected for a free iPhone 15.",
    "You are our 1,000,000th visitor, click to claim your prize now.",
    "Send 0.1 BTC to this address and get 1.0 BTC back immediately.",
    "Earn $500 an hour working from home with no experience required.",
    "Your tax refund is waiting, click this link to verify your bank details.",
    "Claim your $1,000 Amazon gift card before the timer runs out!",
    "Exclusive: Elon Musk is giving away 5,000 Dogecoin to all followers.",
    "Final notice: Your inheritance of $10M is ready for transfer.",
    "Double your money in 24 hours with our guaranteed crypto bot.",
    # ... (Imagine the rest of your 100 sentences are here)
]

# Combine them into one master list
tweets = real_tweets + fake_tweets

# AUTOMATIC LABELS: 1 for Real, 0 for Fake
# This ensures the lengths always match
labels = ([1] * len(real_tweets)) + ([0] * len(fake_tweets))

# -------------------------------
# 2. PROCESSING & TRAINING
# -------------------------------
def clean_text(text):
    text = text.lower()
    text = "".join([c for c in text if c not in string.punctuation])
    return text

cleaned = [clean_text(t) for t in tweets]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(cleaned)

model = LogisticRegression()
model.fit(X, labels)

# -------------------------------
# 3. STREAMLIT UI
# -------------------------------
st.title("🐦 Fake News Detection on Twitter")
st.write("Enter a tweet to analyze whether it is **REAL** or **FAKE**")

user_input = st.text_area("Enter Tweet")

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

        # Analysis Module
        st.subheader("🔍 Analysis")
        st.write(f"Tweet Length: {len(user_input.split())} words")

        fake_keywords = [
            "click", "win", "free", "lottery", "shocking", "money", "cash", 
            "prize", "crypto", "bitcoin", "giveaway", "bonus", "urgent"
            # ... (Add the rest of your 70+ keywords here)
        ]

        detected_flags = [word for word in fake_keywords if word in cleaned_input.split()]

        if detected_flags:
            st.warning(f"⚠️ Suspicious Words Detected: {', '.join(detected_flags)}")
        else:
            st.info("No suspicious keywords detected")

    else:
        st.warning("Please enter a tweet")
