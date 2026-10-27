"""Fake News Detector: a Streamlit page that checks whether an article looks real or fake.

Run from the project folder with:  streamlit run app/app.py
"""

from datetime import datetime

import pandas as pd
import streamlit as st

from preprocess import clean_text
from utils import (
    MAX_EXPLAIN_WORDS,
    MIN_WORDS,
    TOP_WORDS_PATH,
    FetchError,
    explain,
    fetch_article,
    footer_html,
    hero_html,
    highlighted_text_html,
    history_item_html,
    load_css,
    load_model,
    load_model_info,
    load_samples,
    make_features,
    metrics_html,
    notice_html,
    predict_features,
    result_card_html,
    safe,
    sample_text,
    top_words_html,
    verdict,
    word_count,
)

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="wide")

SAMPLES = load_samples()


def html(content):
    """Show a piece of our own HTML."""
    st.markdown(content, unsafe_allow_html=True)


# ---------- Session state ----------

def init_state():
    """Create the values we keep between reruns, the first time the page loads."""
    defaults = {
        "history": [],        # every check, oldest first
        "current_id": None,   # which check the result area shows
        "next_id": 1,
        "article_text": "",   # contents of the text area
        "sample_index": None, # which demo article was loaded last
        "fetched": None,      # last article fetched from a URL
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def use_sample(index):
    """Fill the text area with a demo article."""
    st.session_state.article_text = sample_text(SAMPLES[index])
    st.session_state.sample_index = index


def show_entry(entry_id):
    """Show an older check from the history."""
    st.session_state.current_id = entry_id


def clear_history():
    """Remove all checks."""
    st.session_state.history = []
    st.session_state.current_id = None


def current_entry():
    """The check the result area should show, if any."""
    for entry in st.session_state.history:
        if entry["id"] == st.session_state.current_id:
            return entry
    return None


# ---------- Checking an article ----------

def run_check(text, source, title=None, true_label=None):
    """Predict and explain one article, showing each step, then save it in the history."""
    with st.status("Checking the article...", expanded=True) as status:
        st.write("Cleaning text")
        cleaned = clean_text(text)
        st.write("Converting to TF-IDF features")
        features = make_features(cleaned)
        st.write("Predicting")
        fake_prob = predict_features(features)
        st.write("Explaining the result")
        weights = explain(text)
        status.update(label="Check complete", state="complete", expanded=False)

    entry = {
        "id": st.session_state.next_id,
        "time": datetime.now().strftime("%H:%M:%S"),
        "source": source,
        "title": title or text[:60].replace("\n", " ") + ("…" if len(text) > 60 else ""),
        "text": text,
        "fake_prob": fake_prob,
        "weights": weights,
        "true_label": true_label,
    }
    st.session_state.history.append(entry)
    st.session_state.current_id = entry["id"]
    st.session_state.next_id += 1


def check_pasted_text():
    """Check the text in the text area (pasted or a demo article)."""
    text = st.session_state.article_text.strip()
    if not text:
        html(notice_html("Paste an article first, or try one of the samples."))
        return
    if word_count(text) < MIN_WORDS:
        html(notice_html(
            f"This text has only {word_count(text)} words. Please paste at least {MIN_WORDS} words "
            "so the model has enough writing to judge."
        ))
        return

    # It counts as a sample only if the demo text was not edited
    index = st.session_state.sample_index
    if index is not None and text == sample_text(SAMPLES[index]).strip():
        run_check(text, "Sample", SAMPLES[index]["title"], SAMPLES[index]["label"])
    else:
        run_check(text, "Pasted")


def check_url(url):
    """Fetch an article from a URL and check it."""
    if not url.strip():
        html(notice_html("Enter the address of a news article first."))
        return
    try:
        with st.spinner("Fetching the page..."):
            article = fetch_article(url)
    except FetchError as error:
        st.session_state.fetched = None
        html(notice_html(str(error)))
        return

    st.session_state.fetched = article
    text = f"{article['title']}\n\n{article['text']}".strip()
    run_check(text, "URL", article["title"] or article["url"])


# ---------- Page sections ----------

def paste_tab():
    st.text_area(
        "Article text",
        key="article_text",
        height=220,
        placeholder="Paste the headline and text of a news article here...",
    )
    clicked = st.button("Check article", type="primary", key="check_paste")

    st.caption("Try a sample:")
    columns = st.columns(len(SAMPLES))
    for index, (column, sample) in enumerate(zip(columns, SAMPLES)):
        short_title = sample["title"][:38] + ("…" if len(sample["title"]) > 38 else "")
        column.button(
            short_title, key=f"sample_{index}", on_click=use_sample, args=(index,),
            help=sample["title"], width="stretch",
        )

    if clicked:
        check_pasted_text()


def url_tab():
    url = st.text_input("Article URL", placeholder="https://www.example.com/news/article")
    if st.button("Fetch & check", type="primary", key="check_url"):
        check_url(url)

    fetched = st.session_state.fetched
    if fetched:
        html(f'<div class="fn-fetched">Fetched: <b>{safe(fetched["title"] or fetched["url"])}</b></div>')
        with st.expander("Preview fetched text"):
            words = fetched["text"].split(" ")
            preview = " ".join(words[:300]) + (" …" if len(words) > 300 else "")
            html(f'<div class="fn-preview">{safe(preview)}</div>')


def result_section(entry):
    html('<div class="fn-section-title">Result</div>')
    if entry["id"] != st.session_state.history[-1]["id"]:
        st.caption(f"Showing an earlier check from your history ({entry['time']}).")
    html(result_card_html(entry))

    html('<div class="fn-section-title">Why this result?</div>')
    if word_count(entry["text"]) > MAX_EXPLAIN_WORDS:
        st.caption(f"The explanation is based on the first {MAX_EXPLAIN_WORDS:,} words of this long article.")
    left, right = st.columns([3, 2], gap="large")
    with left:
        html(highlighted_text_html(entry["text"], entry["weights"]))
    with right:
        html(top_words_html(entry["weights"]))


def history_sidebar():
    with st.sidebar:
        st.header("History")
        history = st.session_state.history
        if not history:
            st.caption("Your checks will appear here.")
            return

        st.caption(f"{len(history)} check{'s' if len(history) != 1 else ''} this session")
        rows = [
            {
                "time": e["time"],
                "source": e["source"],
                "title": e["title"],
                "verdict": verdict(e["fake_prob"])[0],
                "confidence_%": round(verdict(e["fake_prob"])[1] * 100, 1),
            }
            for e in history
        ]
        st.button("Clear history", on_click=clear_history, width="stretch")
        st.download_button(
            "Download history as CSV",
            data=pd.DataFrame(rows).to_csv(index=False),
            file_name="fake_news_history.csv",
            mime="text/csv",
            width="stretch",
        )

        # Newest first
        for entry in reversed(history):
            is_current = entry["id"] == st.session_state.current_id
            html(history_item_html(entry, is_current))
            st.button(
                "Show result", key=f"show_{entry['id']}", on_click=show_entry,
                args=(entry["id"],), width="stretch", disabled=is_current,
            )


def about_section(info):
    with st.expander("About this model"):
        st.markdown(
            "**Dataset:** WELFake, about 72,000 news articles (label 1 = fake, 0 = real). "
            "Publisher names, datelines and image credits were removed first, so the model "
            "can't cheat by recognising the source.\n\n"
            "**Preprocessing:** lowercase, remove URLs, HTML and non-letters, remove stopwords "
            "(but keep negations like *not* and *don*), drop very short words, and lemmatize. "
            "The app uses exactly the same `clean_text` function as training.\n\n"
            "**Model:** TF-IDF features (single words and word pairs, 50,000 terms) with a Linear SVM, "
            "calibrated so it can give a probability for the confidence %."
        )
        if info:
            st.markdown(f"**Test-set scores** ({info['test_rows']:,} articles never seen in training):")
            html(metrics_html(info))
        if TOP_WORDS_PATH.exists():
            st.image(str(TOP_WORDS_PATH), caption="Words the model links most with fake and real news")


# ---------- Page ----------

html(load_css())
init_state()
info = load_model_info()
html(hero_html(info))

if load_model() is None:
    html(notice_html(
        "The model file models/fake_news_model.joblib was not found. "
        "Run the Save Final Model cell in notebooks/02_classical_models.ipynb first."
    ))
    html(footer_html())
    st.stop()

paste, from_url = st.tabs(["Paste article", "Check from URL"])
with paste:
    paste_tab()
with from_url:
    url_tab()

entry = current_entry()
if entry:
    result_section(entry)

about_section(info)
html(footer_html())

# Drawn last, so it already includes a check made during this run
history_sidebar()
