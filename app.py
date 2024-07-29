# Library imports
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
import joblib
import string
from flask import Flask, request, jsonify, render_template
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import pickle

# Ensure the VADER lexicon is downloaded
nltk.download('vader_lexicon')

# Create the app object
app = Flask(__name__)

# Load the SentimentIntensityAnalyzer object from the pickle file
with open('vader_sentiment_analyzer.pkl', 'rb') as file:
    sia = pickle.load(file)




# Define predict function
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    new_review = [str(x) for x in request.form.values()]
    text = new_review[0]
#     data = pd.DataFrame(new_review)
#     data.columns = ['new_review']

    predictions = sia.polarity_scores(text)
    compound_score = predictions['compound']
    if compound_score >= 0.05:
        sentiment = 'Positive'
    elif compound_score <= -0.05:
        sentiment = 'Negative'
    else:
        sentiment = 'Neutral'
    
    return render_template('index.html', prediction_text=f'Sentiment: {sentiment}', compound_score=f'Compound Score: {compound_score:.2f}')


if __name__ == "__main__":
    app.run(debug=True)
