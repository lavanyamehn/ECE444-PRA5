from flask import Flask, jsonify, request
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

application = Flask(__name__)

### model loading ###
loaded_model = None
with open('basic_classifier.pkl', 'rb') as fid:
    loaded_model = pickle.load(fid)

vectorizer = None
with open('count_vectorizer.pkl', 'rb') as vd:
    vectorizer = pickle.load(vd)

@application.route('/', methods=['GET'])
def index():
    return "Fake News Detection API, use /predict endpoint."

@application.route('/predict', methods=['POST'])
def predict():
    data = (request.json).get('text', [])

    if not data or not isinstance(data, list):
        return jsonify({'error': 'Need to provide input text for prediction'}), 400

    predictions = []
    for text in data:
        print(text)
        predictions.append(str(loaded_model.predict(vectorizer.transform([text]))))

    return jsonify({'prediction': predictions}), 200

if __name__ == '__main__':
    application.run()