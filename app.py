from flask import Flask, request, jsonify, render_template
from pathlib import Path
import os
import pickle
import re
import warnings
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Initialize the Flask app
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

try:
    stop_words = set(stopwords.words("english"))
except LookupError:
    warnings.warn(
        "NLTK stopwords data is missing; using scikit-learn's built-in "
        "English stopword list.",
        RuntimeWarning,
    )
    stop_words = set(ENGLISH_STOP_WORDS)

# Load models and vectorizer
with (
    (MODEL_DIR / "logistic_regression_model.sav").open("rb") as model_file,
    (MODEL_DIR / "naive_bayes_model.sav").open("rb") as model_file_nb,
    (MODEL_DIR / "vectorizer.pkl").open("rb") as vectorizer_file,
):
    lr_model = pickle.load(model_file)
    nb_model = pickle.load(model_file_nb)
    vectorizer = pickle.load(vectorizer_file)

# Initialize preprocessor components
port_stem = PorterStemmer()

# Preprocess the text
def preprocess_text(content):
    # Remove non-alphabetic characters
    content = re.sub('[^a-zA-Z]', ' ', content)
    # Convert to lowercase and tokenize
    content = content.lower().split()
    # Remove stopwords and apply stemming
    content = [port_stem.stem(word) for word in content if word not in stop_words]
    return ' '.join(content)

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    # Get user input and selected model
    user_text = request.form.get("text", "").strip()
    selected_model = request.form.get("model", "logistic_regression")

    if not user_text:
        return jsonify({"error": "No text provided"}), 400

    models = {
        "logistic_regression": lr_model,
        "naive_bayes": nb_model,
    }
    model = models.get(selected_model)
    if model is None:
        return jsonify({"error": "Invalid model selected"}), 400

    # Preprocess the input
    processed_text = preprocess_text(user_text)
    vectorized_text = vectorizer.transform([processed_text])

    # Predict sentiment
    sentiment = model.predict(vectorized_text)[0]
    probabilities = model.predict_proba(vectorized_text)[0]
    positive_prob = probabilities[1] * 100
    negative_prob = probabilities[0] * 100

    # Map sentiment to human-readable form
    sentiment_label = "Positive" if sentiment == 1 else "Negative"

    # Return JSON response
    return jsonify({
        "model": selected_model,
        "sentiment": sentiment_label,
        "positive_score": round(positive_prob, 2),
        "negative_score": round(negative_prob, 2),
    })

# Feedback route
@app.route('/feedback', methods=['POST'])
def feedback():
    user_feedback = request.form.get("feedback", "").strip()
    if not user_feedback:
        return jsonify({"error": "No feedback provided"}), 400

    # Append feedback to a log file
    with (BASE_DIR / "feedback_log.txt").open("a", encoding="utf-8") as feedback_file:
        feedback_file.write(user_feedback + "\n")

    return jsonify({"message": "Feedback received successfully!"})

# Run the app
if __name__ == '__main__':
    app.run(debug=os.getenv("FLASK_DEBUG") == "1")
