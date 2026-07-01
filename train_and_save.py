import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

emails = []
labels = []

# Load Ham emails
for folder in ["outer_easy_ham", "outer_hard_ham"]:
    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        try:
            with open(path, "r", encoding="latin1") as f:
                emails.append(f.read())
                labels.append(0)
        except:
            pass

# Load Spam emails
for file in os.listdir("outer_spam_2"):
    path = os.path.join("outer_spam_2", file)
    try:
        with open(path, "r", encoding="latin1") as f:
            emails.append(f.read())
            labels.append(1)
    except:
        pass

# TF-IDF
vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(emails)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

# Save BOTH model and vectorizer
with open("spam_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("\nModel and vectorizer saved successfully!")