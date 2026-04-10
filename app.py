import streamlit as st
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Sample Twitter-like data
tweets = [
    "Government announces new policy",
    "Official news released today",
    "Click here to win money now",
    "Fake celebrity scandal click now",
    "Ministry confirms update",
    "You won lottery claim prize"
]

labels = [1, 1, 0, 0, 1, 0]

# Simple clean (no nltk)
def clean_text(text):
    text = text.lower()
    text = "".join([c for c in text if c not in string.punctuation])
    return text

# Train model
cleaned = [clean_text(t) for t in tweets]
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(cleaned)

model = LogisticRegression()
model.fit(X, labels)

# UI
st.title("🐦 Fake News Detection on Twitter")

user_input = st.text_area("Enter Tweet")

if st.button("Analyze"):
    if user_input.strip() != "":
        cleaned_input = clean_text(user_input)
        vector = vectorizer.transform([cleaned_input])
        prediction = model.predict(vector)

        if prediction[0] == 1:
            st.success("✅ Real Tweet")
        else:
            st.error("❌ Fake Tweet")
    else:
        st.warning("Enter some text")
