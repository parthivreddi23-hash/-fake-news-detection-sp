import streamlit as st
import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Download stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords

# Sample Twitter-like data
tweets = [
    "Government announces new economic policy today",
    "Breaking: official statement released by PMO",
    "Click here to win free iPhone now!!!",
    "Shocking celebrity scandal revealed click now",
    "Ministry confirms new education reforms",
    "You won lottery claim your prize immediately"
]

labels = [1, 1, 0, 0, 1, 0]  # 1 = Real, 0 = Fake

# Clean text
def clean_text(text):
    text = text.lower()
    text = "".join([c for c in text if c not in string.punctuation])
    words = text.split()
    words = [w for w in words if w not in stopwords.words('english')]
    return " ".join(words)

# Train model
cleaned = [clean_text(t) for t in tweets]
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(cleaned)

model = LogisticRegression()
model.fit(X, labels)

# Streamlit UI
st.title("🐦 Fake News Detection on Twitter")

st.write("Analyze tweets and detect whether they are REAL or FAKE")

user_input = st.text_area("Enter Tweet")

if st.button("Analyze Tweet"):
    if user_input.strip() != "":
        cleaned_input = clean_text(user_input)
        vector = vectorizer.transform([cleaned_input])
        prediction = model.predict(vector)

        if prediction[0] == 1:
            st.success("✅ Real Tweet")
        else:
            st.error("❌ Fake Tweet")
    else:
        st.warning("Please enter a tweet")
