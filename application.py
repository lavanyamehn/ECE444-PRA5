from flask import Flask
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

### How to use model to predict
prediction = loaded_model.predict(vectorizer.transform(['This is fake news']))[0]

@application.route('/', methods=['GET', 'POST'])
def index():
    return prediction

if __name__ == '__main__':
    application.run()