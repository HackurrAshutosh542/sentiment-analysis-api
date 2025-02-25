from flask import Flask, request, jsonify
import joblib

# ✅ Load trained models
tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")
sentiment_model = joblib.load("sentiment_model.pkl")

app = Flask(__name__)

@app.route("/predict_sentiment", methods=["POST"])
def predict_sentiment():
    data = request.json
    text = data["text"]
    
    # Convert text to TF-IDF format
    text_tfidf = tfidf_vectorizer.transform([text])
    
    # Predict sentiment
    prediction = sentiment_model.predict(text_tfidf)[0]
    sentiment_label = { -1: "Negative", 1: "Positive" }[prediction]
    
    return jsonify({"text": text, "predicted_sentiment": sentiment_label})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
