import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os 

# Step 1: Load the dataset
df = pd.read_csv(r"C:\Ayush\Coding\Python\phishing_email_detector\data\phishing_dataset.csv")
print(df.columns)

df["combined"] = df["subject"] + " " + df["body"]
X = df["combined"]
y = df["label"]

# Step 2: Vectorize text (convert words to numbers)
X = X.fillna("")  # Replace NaN with empty strings
vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
X_vec = vectorizer.fit_transform(X)

# Step 3: Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42
)

# Step 4: Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 5: Evaluate the model
y_pred = model.predict(X_test)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("🔍 Report:\n", classification_report(y_test, y_pred))

os.makedirs("models", exist_ok=True)

# Step 6: Save the model and vectorizer
joblib.dump(model, "models/phishing_model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")

print("✅ Model and vectorizer saved successfully.")
