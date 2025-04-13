# app/predict.py
import joblib
import os

# Load model and vectorizer
model = joblib.load("models/phishing_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")  # if you saved it

def predict_email(subject, body):
    combined_text = subject + " " + body
    email_vec = vectorizer.transform([combined_text])
    prediction = model.predict(email_vec)[0]
    return "Phishing" if prediction == 1 else "Legitimate"


# Example usage
if __name__ == "__main__":
    subject=input("Paste Subject : ")
    email = input("Paste email content: ")
    print("Prediction:", predict_email(subject, email))
