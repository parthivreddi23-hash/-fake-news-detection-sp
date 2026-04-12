from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from utils import clean_text

# sample data
tweets = [
    "Government announces new policy",
    "Official news released today",
    "Click here to win money now",
    "Fake celebrity scandal click now",
    "Ministry confirms update",
    "You won lottery claim prize"
]

labels = [1, 1, 0, 0, 1, 0]

# preprocessing
cleaned = [clean_text(t) for t in tweets]

# vectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(cleaned)

# model
model = LogisticRegression()
model.fit(X, labels)

def predict(text):
    cleaned = clean_text(text)
    vector = vectorizer.transform([cleaned])
    pred = model.predict(vector)
    prob = model.predict_proba(vector)
    return pred[0], max(prob[0])
