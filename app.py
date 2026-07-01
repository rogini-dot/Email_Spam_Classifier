from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)

# Load model and vectorizer
with open("spam_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    email_text = data.get("email", "")

    if not email_text.strip():
        return jsonify({"result": "error", "message": "Please enter some text!"})

    vec = vectorizer.transform([email_text])
    prediction = model.predict(vec)[0]

    if prediction == 1:
        return jsonify({"result": "spam", "message": "SPAM EMAIL"})
    else:
        return jsonify({"result": "normal", "message": "NORMAL EMAIL"})

if __name__ == "__main__":
    app.run(debug=True)