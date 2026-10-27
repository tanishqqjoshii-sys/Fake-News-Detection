# Fake News Detection using NLP

Classifying news articles as real or fake using NLP techniques on the WELFake dataset.

## How to run the app

You need Python 3.12 or newer. Run these commands from the project folder (the folder name has spaces, so keep the quotes).

1. Create and activate a virtual environment:

   ```bash
   cd "Fake News Detection"
   python3 -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate
   ```

2. Install the packages:

   ```bash
   pip install -r requirements.txt
   ```

3. Download `WELFake_Dataset.csv` and put it in the `data/` folder.

4. Run both notebooks from top to bottom, in order:
   - `notebooks/01_eda.ipynb` cleans the data and creates `train.csv` and `test.csv`.
   - `notebooks/02_classical_models.ipynb` trains and evaluates the models, and its last cell saves `models/fake_news_model.joblib`.

5. Start the app:

   ```bash
   streamlit run app/app.py
   ```

   It opens in your browser at http://localhost:8501. Paste an article, try one of the samples, or check an article from a URL.
