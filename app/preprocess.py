"""Text cleaning shared by the notebooks and the Streamlit app."""

import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

for resource in ["stopwords", "wordnet", "omw-1.4"]:
    nltk.download(resource, quiet=True)

# Negations flip meaning; "don", "isn" etc. lost their apostrophes
NEGATIONS = {
    "not", "no", "nor",
    "don", "didn", "doesn", "isn", "wasn", "aren", "weren", "won", "wouldn",
    "shouldn", "couldn", "hasn", "haven", "hadn", "ain", "mustn", "needn",
}

STOP_WORDS = set(stopwords.words("english")) - NEGATIONS
LEMMATIZER = WordNetLemmatizer()


def clean_text(text: str) -> str:
    """Lowercase, remove noise and stopwords, and lemmatize each word."""
    # Missing text (None or NaN) becomes empty
    if not isinstance(text, str):
        text = ""

    text = text.lower()

    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)

    text = re.sub(r"[^a-z\s]", " ", text)

    words = []
    for word in text.split():
        if word in STOP_WORDS or len(word) <= 2:
            continue
        words.append(LEMMATIZER.lemmatize(word))

    return " ".join(words)
