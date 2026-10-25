"""Text cleaning shared by the notebooks and the Streamlit app."""

import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download the NLTK data we need (skipped quickly if already present)
for resource in ["stopwords", "wordnet", "omw-1.4"]:
    nltk.download(resource, quiet=True)

# Negation words to keep, because they can flip the meaning of a sentence.
# The dataset has no apostrophes ("don't" became "don t"), so we also keep
# the first part of contractions like "don", "isn" and "wasn".
NEGATIONS = {
    "not", "no", "nor",
    "don", "didn", "doesn", "isn", "wasn", "aren", "weren", "won", "wouldn",
    "shouldn", "couldn", "hasn", "haven", "hadn", "ain", "mustn", "needn",
}

# Build these once so clean_text stays fast
STOP_WORDS = set(stopwords.words("english")) - NEGATIONS
LEMMATIZER = WordNetLemmatizer()


def clean_text(text: str) -> str:
    """Lowercase, remove noise and stopwords, and lemmatize each word."""
    text = text.lower()

    # Remove URLs and HTML tags
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)

    # Keep only letters and spaces
    text = re.sub(r"[^a-z\s]", " ", text)

    # Split into words, drop stopwords and very short words, then lemmatize
    words = []
    for word in text.split():
        if word in STOP_WORDS or len(word) <= 2:
            continue
        words.append(LEMMATIZER.lemmatize(word))

    return " ".join(words)


if __name__ == "__main__":
    examples = [
        "BREAKING: Scientists DON'T want you to know this! Read more at https://example.com <b>now</b>",
        "The senators were not convinced by the new healthcare bills, officials said on Monday.",
    ]
    for sentence in examples:
        print("Before:", sentence)
        print("After: ", clean_text(sentence))
        print()
