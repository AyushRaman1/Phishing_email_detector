from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import joblib
import os   

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the model and vectorizer
model = joblib.load(os.path.join("models", "phishing_model.pkl"))
vectorizer = joblib.load(os.path.join("models", "vectorizer.pkl"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    subject = request.form['subject']
    content = request.form['content']
    full_email = subject + " " + content

    email_vector = vectorizer.transform([full_email])
    prediction = model.predict(email_vector)[0]

    result = "Phishing Email 🚨" if prediction == 1 else "Safe Email ✅"
    return render_template('index.html', result=result)

@app.route("/predict_api", methods=["POST"])
def predict_api():
    data = request.get_json()
    subject = data.get("subject", "")
    body = data.get("body", "")
    full_email = subject + " " + body

    email_vector = vectorizer.transform([full_email])
    prediction = model.predict(email_vector)[0]

    return jsonify({"prediction": int(prediction)})

if __name__ == '__main__':
    app.run(debug=True)
