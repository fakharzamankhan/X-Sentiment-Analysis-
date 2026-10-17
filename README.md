# Twitter Sentiment Analysis

A Flask web application that classifies text as positive or negative using
TF-IDF features and two trained scikit-learn models:

- Logistic Regression
- Naive Bayes

The application also displays the confidence scores returned by the selected
model and accepts optional local feedback.

## Project structure

```text
.
├── app.py
├── models/
│   ├── logistic_regression_model.sav
│   ├── naive_bayes_model.sav
│   └── vectorizer.pkl
├── static/
│   ├── style.css
│   └── twitter.jpeg
├── templates/
│   └── index.html
└── requirements.txt
```

## Run locally

1. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   ```

   Windows PowerShell:

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

2. Install the dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

3. Download the NLTK stopwords corpus once (optional; the app uses a
   scikit-learn fallback if it is unavailable):

   ```bash
   python -c "import nltk; nltk.download('stopwords')"
   ```

4. Start the application:

   ```bash
   python app.py
   ```

5. Open <http://127.0.0.1:5000> in a browser.

For development-only debug mode, set `FLASK_DEBUG=1` before starting the
application. Debug mode should not be enabled in production.

## API endpoints

### `POST /predict`

Form fields:

- `text`: text to classify
- `model`: `logistic_regression` or `naive_bayes`

Returns the selected model, sentiment label, and positive/negative scores as
JSON.

### `POST /feedback`

Form field:

- `feedback`: feedback text

Feedback is appended to the local `feedback_log.txt` file. This file is
intentionally ignored by Git because it may contain user-submitted content.

## Notes

- The trained model files are included so the demo can run without retraining.
- scikit-learn is pinned to the version used to serialize the bundled models.
- Pickle files should only be loaded from trusted sources.
- This is an educational project, not a production authentication or
  moderation system.
