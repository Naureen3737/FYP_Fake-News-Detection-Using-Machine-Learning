from flask import Flask, request, jsonify, render_template, send_from_directory, redirect
import requests
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os
import pandas as pd
from sklearn.linear_model import PassiveAggressiveClassifier
import nltk

# Ensure required NLTK corpora are downloaded
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()



# Initialize app
app = Flask(__name__)

# Download necessary NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('omw-1.4', quiet=True)

# Load model
try:
    with open("pac_model.pkl", "rb") as f:
        pac_model = pickle.load(f)
    print("✅ PassiveAggressiveClassifier model loaded")
except Exception as e:
    print(f"❌ Model load error: {e}")
    pac_model = None

# Load vectorizer
try:
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    print("✅ Vectorizer loaded")
except Exception as e:
    print(f"❌ Vectorizer load error: {e}")
    vectorizer = None

# Preprocess function
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    """Enhanced text cleaning"""
    if not isinstance(text, str):
        return ""
    # Handle encoding artifacts
    text = text.encode('ascii', 'ignore').decode('ascii')
    
    # Standard cleaning
    text = re.sub(r'[^a-zA-Z\s]', ' ', text.lower())
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word.strip()) 
             for word in tokens 
             if word not in stop_words and len(word) > 2]
    return ' '.join(tokens)

# Home
@app.route('/')
def home():
    return render_template("index.html")

# Prediction page
@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if request.method == 'POST':
        news = request.form.get('news')
        if news:
            cleaned_news = preprocess(news)
            if not cleaned_news.strip():
                return render_template('prediction.html', prediction_text="Input too short or invalid.")
            if pac_model is None or vectorizer is None:
                return render_template('prediction.html', prediction_text="Model or vectorizer not loaded.")
            prediction_label = pac_model.predict(vectorizer.transform([cleaned_news]))[0]
            prediction_text = "REAL" if prediction_label == "REAL" else "FAKE"
            return render_template('prediction.html', prediction_text=prediction_text, headline=news)
        else:
            return render_template('prediction.html', prediction_text="Please enter a news headline.")
    return render_template('prediction.html')  # Handle GET request

# Real-time news
NEWS_API_KEY = '7359c2b45b4442fca0ca5107eb43e10d'

@app.route('/live-news')
def live_news():
    headlines_response = requests.get('https://newsapi.org/v2/top-headlines', params={'country': 'us', 'apiKey': NEWS_API_KEY})
    if headlines_response.status_code != 200:
        return render_template("error.html", error_message=f"News API failed: {headlines_response.status_code}")

    articles = headlines_response.json().get("articles", [])
    results = []

    for article in articles[:10]:
        title = article.get("title", "")
        cleaned = preprocess(title)
        if cleaned.strip() and pac_model and vectorizer:
            prediction = pac_model.predict(vectorizer.transform([cleaned]))[0]
        else:
            prediction = "Unknown"
        results.append({
            "title": title,
            "source": article.get("source", {}).get("name", "Unknown"),
            "url": article.get("url", ""),
            "description": article.get("description", "No description."),
            "date": article.get("publishedAt", "Unknown"),
            "prediction": prediction
        })

    return render_template('live_news.html', results=results)

# Simple API endpoint for AJAX prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Handle both JSON and form data
    if request.is_json:
        data = request.get_json()
        news_content = data.get('newsContent', '')
    else:
        news_content = request.form.get('newsContent', '')
    
    if not news_content:
        return jsonify({"error": "No content provided"}), 400

    cleaned = preprocess(news_content)
    if not cleaned.strip():
        return jsonify({"error": "Input too short or invalid"}), 400
    
    if pac_model is None or vectorizer is None:
        return jsonify({"error": "Model or vectorizer not loaded"}), 500

    prediction = pac_model.predict(vectorizer.transform([cleaned]))[0]
    return jsonify({"prediction": "REAL" if prediction == "REAL" else "FAKE"})

# Submit feedback
@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    headline = request.form.get('headline')
    prediction = request.form.get('prediction')
    feedback = request.form.get('feedback')

    feedback_data = {'headline': headline, 'prediction': prediction, 'feedback': feedback}
    feedback_df = pd.DataFrame([feedback_data])
    feedback_file = 'feedback.csv'

    if os.path.exists(feedback_file):
        feedback_df.to_csv(feedback_file, mode='a', header=False, index=False)
    else:
        feedback_df.to_csv(feedback_file, index=False)

    return redirect('/')

# Serve favicon
@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico', mimetype='image/vnd.microsoft.icon')

# Error pages
@app.errorhandler(404)
def not_found_error(e):
    return render_template("error.html", error_message="Page not found."), 404

@app.errorhandler(500)
def internal_error(e):
    return render_template("error.html", error_message="Internal server error."), 500

# Run
if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)