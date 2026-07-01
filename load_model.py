import pickle

# Load model
loaded_model = pickle.load(open("spam_model.pkl", "rb"))

print("Model loaded successfully!")