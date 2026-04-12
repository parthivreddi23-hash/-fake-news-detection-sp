import string

def clean_text(text):
    text = text.lower()
    text = "".join([c for c in text if c not in string.punctuation])
    return text
