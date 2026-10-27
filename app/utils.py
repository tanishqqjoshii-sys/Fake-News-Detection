"""Helpers for the Streamlit app: loading files, predicting, fetching URLs, LIME and HTML pieces."""

import html
import json
import re
from pathlib import Path

import joblib
import requests
import streamlit as st
from bs4 import BeautifulSoup
from lime.lime_text import LimeTextExplainer

from preprocess import clean_text

# File locations, based on where this file lives (not where the app was started)
APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent
MODEL_PATH = PROJECT_DIR / "models" / "fake_news_model.joblib"
INFO_PATH = PROJECT_DIR / "models" / "model_info.json"
SAMPLES_PATH = APP_DIR / "samples.json"
CSS_PATH = APP_DIR / "style.css"
TOP_WORDS_PATH = PROJECT_DIR / "reports" / "figures" / "top_words.png"

MIN_WORDS = 20            # shorter texts don't give the model enough to work with
MAX_EXPLAIN_WORDS = 1000  # LIME only looks at the start of very long articles, to stay fast

# A normal browser User-Agent, because many news sites block unknown clients
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


# ---------- Loading files ----------

@st.cache_resource
def load_model():
    """Load the saved model once. Returns None if the file is missing."""
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


def load_json(path, default):
    """Read a JSON file, or return the default if it does not exist."""
    if not path.exists():
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_model_info():
    """Model name, test scores and dataset sizes saved by the notebook."""
    return load_json(INFO_PATH, {})


def load_samples():
    """The four demo articles."""
    return load_json(SAMPLES_PATH, [])


def load_css():
    """The page styles as a <style> tag."""
    return f"<style>{CSS_PATH.read_text(encoding='utf-8')}</style>"


def word_count(text):
    """Number of words in a text."""
    return len(text.split())


# ---------- Prediction ----------

def make_features(cleaned_text):
    """Turn cleaned text into a TF-IDF vector."""
    return load_model().named_steps["tfidf"].transform([cleaned_text])


def predict_features(features):
    """Probability that the article is fake (0 to 1)."""
    return float(load_model().named_steps["clf"].predict_proba(features)[0, 1])


def predict(text):
    """Probability that a raw article is fake: clean, convert to TF-IDF, predict."""
    return predict_features(make_features(clean_text(text)))


def verdict(fake_prob):
    """The predicted label and how confident the model is in it."""
    if fake_prob >= 0.5:
        return "Fake", fake_prob
    return "Real", 1 - fake_prob


# ---------- Fetching an article from a URL ----------

class FetchError(Exception):
    """A problem with a URL, with a message that is safe to show to the user."""


def fetch_article(url):
    """Download a web page and return its title and paragraph text."""
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
    except requests.exceptions.Timeout:
        raise FetchError("The page took too long to respond. Try again, or paste the article text instead.")
    except requests.exceptions.RequestException:
        raise FetchError("Could not reach this page. Check the URL, or paste the article text instead.")

    if response.status_code in (401, 402, 403):
        raise FetchError("This site blocked the request or needs a subscription. Paste the article text instead.")
    if response.status_code >= 400:
        raise FetchError(f"The page returned an error (code {response.status_code}). Check the URL.")
    if "html" not in response.headers.get("Content-Type", ""):
        raise FetchError("This link does not point to a web page.")

    soup = BeautifulSoup(response.content, "html.parser")
    title = soup.title.get_text(" ", strip=True) if soup.title else ""

    # Keep paragraphs with at least 5 words, to skip menus, buttons and captions
    paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
    paragraphs = [p for p in paragraphs if word_count(p) >= 5]
    text = "\n\n".join(paragraphs)

    if word_count(text) < MIN_WORDS:
        raise FetchError(
            f"Found fewer than {MIN_WORDS} words of article text. The page may be paywalled "
            "or built with JavaScript. Paste the article text instead."
        )
    return {"title": title, "text": text, "url": url}


# ---------- Explanation with LIME ----------

@st.cache_resource
def get_explainer():
    """Create the LIME explainer once."""
    return LimeTextExplainer(class_names=["Real", "Fake"], random_state=42)


def predict_many(texts):
    """[P(real), P(fake)] for a list of raw texts. LIME calls this many times."""
    return load_model().predict_proba([clean_text(text) for text in texts])


def explain(text):
    """The 10 words that pushed the prediction most, as (word, weight) pairs.
    A positive weight pushes towards Fake, a negative weight towards Real."""
    short_text = " ".join(text.split()[:MAX_EXPLAIN_WORDS])
    explanation = get_explainer().explain_instance(
        short_text, predict_many, num_features=10, num_samples=500
    )
    return [(str(word), float(weight)) for word, weight in explanation.as_list()]


# ---------- HTML pieces ----------

def safe(text):
    """Escape text for HTML (also "$", which Streamlit would read as maths)."""
    return html.escape(text).replace("$", "&#36;").replace("\n", "<br>")


def sample_text(sample):
    """A demo article as it appears in the text box: title line, then the text."""
    return f"{sample['title']}\n\n{sample['text']}"


def hero_html(info):
    """Header card with the title, subtitle and stat chips."""
    chips = ""
    if info:
        chip_texts = [
            f"{info['f1'] * 100:.1f}% F1",
            f"~{round(info['train_rows'] / 1000)}k training articles",
            info["model_name"],
        ]
        chips = "".join(f'<span class="fn-chip">{safe(c)}</span>' for c in chip_texts)
    return (
        '<div class="fn-hero">'
        '<div class="fn-hero-title">Fake News Detector</div>'
        '<div class="fn-hero-sub">Check whether a news article looks real or fake using NLP</div>'
        f'<div class="fn-chips">{chips}</div>'
        "</div>"
    )


def notice_html(message):
    """A friendly message box in the page colours."""
    return f'<div class="fn-notice">{safe(message)}</div>'


def meter_html(fake_prob):
    """Animated bar from Real to Fake with a marker at the fake probability."""
    percent = round(fake_prob * 100, 1)
    return (
        '<div class="fn-meter">'
        '<div class="fn-meter-labels"><span>Real</span><span>Fake</span></div>'
        '<div class="fn-meter-track">'
        f'<div class="fn-meter-fill" style="width: {percent}%"></div>'
        f'<div class="fn-meter-marker" style="left: {percent}%"></div>'
        "</div>"
        f'<div class="fn-meter-caption">Fake probability: {percent:.0f}%</div>'
        "</div>"
    )


def result_card_html(entry):
    """The big verdict card for one check."""
    label, confidence = verdict(entry["fake_prob"])
    if label == "Fake":
        css_class, title = "fake", "⚠︎ Likely FAKE"
    else:
        css_class, title = "real", "✓ Likely REAL"

    uncertain = ""
    if 0.4 <= confidence <= 0.6:
        uncertain = '<div class="fn-uncertain">Uncertain — verify with trusted sources</div>'

    details = [entry["source"], entry["time"]]
    if entry.get("true_label"):
        details.append(f"Dataset label: {entry['true_label']}")

    return (
        f'<div class="fn-card fn-result {css_class}">'
        '<div class="fn-result-top">'
        f'<div><div class="fn-verdict">{title}</div>'
        f'<div class="fn-result-title">{safe(entry["title"])}</div></div>'
        f'<div class="fn-conf"><span>{confidence * 100:.0f}%</span><small>confidence</small></div>'
        "</div>"
        f"{meter_html(entry['fake_prob'])}"
        f"{uncertain}"
        f'<div class="fn-result-meta">{safe(" · ".join(details))}</div>'
        "</div>"
    )


def legend_html():
    """Small legend for the highlight colours."""
    return (
        '<div class="fn-legend">'
        '<span><i class="fn-swatch fake"></i>pushes towards Fake</span>'
        '<span><i class="fn-swatch real"></i>pushes towards Real</span>'
        "</div>"
    )


def highlighted_text_html(text, weights, max_words=250):
    """The start of the article with LIME's top words highlighted.
    Stronger weight = stronger colour."""
    weight_of = dict(weights)
    biggest = max((abs(w) for w in weight_of.values()), default=1) or 1

    parts = []
    words_seen = 0
    # Split into words and the spaces/punctuation between them, keeping both
    for token in re.split(r"(\W+)", text):
        if words_seen >= max_words:
            parts.append(" …")
            break
        if token in weight_of:
            weight = weight_of[token]
            strength = abs(weight) / biggest
            if weight > 0:
                alpha = 0.35 + 0.65 * strength
                style = f"background: rgba(234, 170, 0, {alpha:.2f})"
                parts.append(f'<mark class="fn-hl fake" style="{style}">{safe(token)}</mark>')
            else:
                alpha = 0.6 + 0.4 * strength  # stays dark enough for light text
                style = f"background: rgba(58, 15, 52, {alpha:.2f})"
                parts.append(f'<mark class="fn-hl real" style="{style}">{safe(token)}</mark>')
        else:
            parts.append(safe(token))
        if re.match(r"\w", token):
            words_seen += 1

    return f'<div class="fn-card">{legend_html()}<div class="fn-text">{"".join(parts)}</div></div>'


def top_words_html(weights):
    """Top-10 words as bars: Real to the left of the centre line, Fake to the right."""
    biggest = max((abs(w) for _, w in weights), default=1) or 1
    rows = []
    for word, weight in sorted(weights, key=lambda pair: abs(pair[1]), reverse=True):
        width = abs(weight) / biggest * 100
        bar = f'<div class="fn-bar {"fake" if weight > 0 else "real"}" style="width: {width:.0f}%"></div>'
        left, right = ("", bar) if weight > 0 else (bar, "")
        rows.append(
            '<div class="fn-word-row">'
            f'<span class="fn-word">{safe(word)}</span>'
            f'<div class="fn-half left">{left}</div>'
            f'<div class="fn-half right">{right}</div>'
            f'<span class="fn-weight">{weight:+.3f}</span>'
            "</div>"
        )
    header = (
        '<div class="fn-word-row fn-word-head">'
        '<span class="fn-word"></span><div class="fn-half left">← Real</div>'
        '<div class="fn-half right">Fake →</div><span class="fn-weight">weight</span>'
        "</div>"
    )
    return f'<div class="fn-card"><div class="fn-words">{header}{"".join(rows)}</div></div>'


def history_item_html(entry, is_current):
    """A compact card for one check in the sidebar."""
    label, confidence = verdict(entry["fake_prob"])
    current = " current" if is_current else ""
    return (
        f'<div class="fn-hist{current}">'
        f'<span class="fn-badge {label.lower()}">{label} · {confidence * 100:.0f}%</span>'
        f'<div class="fn-hist-title">{safe(entry["title"])}</div>'
        f'<div class="fn-hist-meta">{safe(entry["time"])} · {safe(entry["source"])}</div>'
        "</div>"
    )


def metrics_html(info):
    """Test-set scores from model_info.json as a row of small stat cards."""
    names = [("Accuracy", "accuracy"), ("Precision", "precision"), ("Recall", "recall"), ("F1", "f1")]
    cards = "".join(
        f'<div class="fn-stat"><span>{info[key] * 100:.1f}%</span><small>{name}</small></div>'
        for name, key in names
    )
    return f'<div class="fn-stats">{cards}</div>'


def footer_html():
    """Project line and a reminder that the model checks style, not facts."""
    return (
        '<div class="fn-footer">'
        "<div>Fake News Detection using NLP · B.Tech CSE Training Project</div>"
        '<div class="fn-footer-note">This model checks writing patterns, not facts. '
        "Always verify important news with trusted sources.</div>"
        "</div>"
    )
