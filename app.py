import streamlit as st
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# -------------------------------
# Sample Twitter Data (Training)
# -------------------------------
tweets = [
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
    "Win a free trip to Hawaii by clicking the link in our bio.",
    "Get a $200 Starbucks voucher by completing this 1-minute survey.",
    "Cash app glitch! Get $100 for free right now.",
    "Become a millionaire overnight using this secret trading loophole.",
    "This teenager made $1M in a week; see how he did it.",
    "Your PayPal account has a pending payment of $500. Click to accept.",
    "Get paid $50 for every tweet you post with this new app.",
    "Free Nitro for everyone! Just click the link below.",
    "We are looking for 10 people to test the new PlayStation 6.",
    "Win a luxury car by liking and sharing this post!",
    "Shocking video: You won't believe what this actress said on set.",
    "10 things doctors don't want you to know about your health.",
    "He thought he found a rock, but what was inside changed his life.",
    "This simple fruit is the secret to never getting sick again.",
    "The hidden truth about the moon landing finally exposed.",
    "You’ve been eating pizza wrong your whole life!",
    "99% of people fail this simple logic test. Are you one of them?",
    "See the photo that the government tried to ban from the internet.",
    "She went to sleep a normal girl and woke up with a superpower.",
    "The real reason this celebrity couple broke up will ruin your day.",
    "Warning: Do not drink water until you see this video.",
    "What happened to this child star is absolutely heartbreaking.",
    "This 5-minute trick will make you look 20 years younger.",
    "They found a giant skeleton in the desert; see the leaked photos.",
    "The world ends tomorrow? Scientists are terrified.",
    "This common household item is actually a deadly poison.",
    "A secret door was found under the Great Pyramid. Click to see inside.",
    "Hollywood is trembling! This whistleblower just told all.",
    "Stop everything you're doing and watch this viral clip.",
    "You won't believe who this famous politician is actually related to.",
    "URGENT: Your account will be deleted in 10 minutes. Click to save.",
    "Someone from Russia just logged into your Twitter account. Verify now.",
    "Action Required: Your Netflix subscription has been suspended.",
    "Unusual activity detected on your credit card. Confirm identity here.",
    "Your Apple ID is locked for security reasons. Click to unlock.",
    "SECURITY ALERT: Change your password immediately via this link.",
    "We found a virus on your phone. Click to scan and remove.",
    "Your package is stuck at the warehouse. Pay $1 to release it.",
    "Official Warning: Your social security number has been flagged.",
    "Click here to see who has been viewing your profile!",
    "Your email storage is 99% full. Click to add 100GB for free.",
    "Someone just posted a private photo of you. See it here.",
    "You have (1) new voicemail. Click to listen.",
    "Facebook is going to start charging users tomorrow. Share to stay free.",
    "Your bank account has been temporarily restricted. Verify info.",
    "Download this update to keep your computer safe from hackers.",
    "A lawsuit has been filed against you. View the documents here.",
    "Your insurance policy is about to expire. Renew now for 90% off.",
    "We detected a suspicious login from a new device. Is this you?",
    "To keep your account active, please log in through our secure portal.",
    "Doctors are speechless: This juice cures cancer in 48 hours.",
    "This one herb can replace all your prescriptions.",
    "Big Pharma doesn't want you to know about this $2 cure.",
    "Lose 30 pounds in a week without exercise or dieting!",
    "Scientists have found the 'God Gene' in a secret lab.",
    "This magnetic bracelet cures arthritis instantly.",
    "A secret plant from the Amazon restores 100% of your vision.",
    "Drinking this before bed burns belly fat while you sleep.",
    "The truth about vaccines that they are hiding from you.",
    "Reverse aging with this ancient Tibetan breathing technique.",
    "Instant hair growth for bald men using this kitchen ingredient.",
    "Eliminate all tooth pain forever with this 10-second ritual.",
    "This pillow cures insomnia and guarantees deep sleep.",
    "Why you should never wear sunscreen, according to a 'real' expert.",
    "The secret frequency that heals the body while you listen.",
    "BREAKING: Aliens have landed in a remote part of Australia.",
    "The government is using weather machines to control the election.",
    "Leaked documents prove the Earth is actually flat.",
    "This famous singer is actually a clone; here is the proof.",
    "A massive asteroid is hitting Earth tonight, but NASA is silent.",
    "The internet will be shut down globally for 10 days. Prepare now!",
    "5G towers are actually mind-control devices. Share before deleted.",
    "Secret underground cities found beneath major world capitals.",
    "This historical figure is actually still alive and living in hiding.",
    "The Great Reset is happening tomorrow; withdraw your cash now.",
    "Evidence found that time travel is real and happening in 2024.",
    "The secret society that runs the world has just been exposed.",
    "Why the sky is blue is actually a giant projection.",
    "Hidden cameras found in all new smart TVs; see how to disable.",
    "The moon is actually a hollow space station.",
    "New law makes it illegal to disagree with the government.",
    "Robots have already replaced 50% of the population.",
    "This city doesn't actually exist; it's a giant movie set.",
    "They are putting chemicals in the water to make us forget the past.",
    "A map of the 'Real World' beyond the ice wall has been found.",
    "Your phone is listening to you even when it's turned off.",
    "This world leader was replaced by an AI three months ago.",
    "Why they are banning gas stoves is part of a darker plan.",
    "The Mandela Effect is proof we are living in a simulation.",
    "They are hiding a second sun behind the clouds."
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
