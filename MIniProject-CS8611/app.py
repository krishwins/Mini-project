import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import string
from flask import Flask, request,render_template,url_for

import joblib
import json
import pickle

import requests

app = Flask (__name__)

@app.route('/') 
def welcome(): 
    return render_template('index.html')

def preprocess(review):
    # Convert to lowercase
    review = review.lower()

    # Tokenize the review text
    tokens = word_tokenize(review)

    # Remove stopwords and punctuation
    stop_words = set(stopwords.words('english'))
    table = str.maketrans('', '', string.punctuation)
    tokens = [w.translate(table) for w in tokens if not w in stop_words]

    # Stem the words
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(w) for w in tokens]

    # Combine the tokens back into a single string
    preprocessed_review = ' '.join(tokens)
    
    return preprocessed_review

# Define the API endpoint for sentiment analysis
@app.route('/predict', methods=['POST'])
def predict():
    # Get the review text from the request data
    review = request.form['review']
    
    # Preprocess the review text
    preprocessed_review = preprocess(review)
    
    # Load the pre-trained tf-idf vectorizer
    with open('tfidf.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
     
    #vectorizer.fit
     
    # Convert the preprocessed review text to a tf-idf feature vector
    features = vectorizer.transform([preprocessed_review])
    
    # Load the pre-trained model
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    # Predict the sentiment of the preprocessed review using the pre-trained model
    sentiment = model.predict(features)[0]
    s=str(sentiment)
    # Return the predicted sentiment as a JSON response
    response = {
        'sentiment': s
    }
    print(sentiment)
    return json.dumps(response)