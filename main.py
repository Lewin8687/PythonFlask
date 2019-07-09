"""
Routes for the flask application.
"""

import os
import json
import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import model_from_json
from flask import request
from flask import Response
from flask import jsonify
from flask import Flask
from flask import render_template
from numpy import array

from DocumentMLClassifier import app

imdb = tf.keras.datasets.imdb
# A dictionary mapping words to an integer index
word_index = imdb.get_word_index()
script_dir = os.path.dirname(__file__)

# The first indices are reserved
word_index = {k:(v+3) for k,v in word_index.items()} 
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def get_model():
  # Model reconstruction from JSON file
  json_file = open(os.path.join(script_dir, 'model/model_architecture.json'),'r')
  loaded_model_json = json_file.read()
  json_file.close()
  model = model_from_json(loaded_model_json)
  # Load weights into new model
  model.load_weights(os.path.join(script_dir, "model/model_weights.h5"))

  print("********Model loaded!")
  return model

def decode_review(text):
  return ' '.join([reverse_word_index.get(i, '?') for i in text])

@app.route('/')
def hello_world():
  return 'Hey its Python Flask application!'

@app.route('/dataset', methods=["GET"])
def predict():
  index = int(request.args["index"])

  (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

  # Preprocess review
  train_data = tf.keras.preprocessing.sequence.pad_sequences(train_data,
                                                    value=word_index["<PAD>"],
                                                    padding='post',
                                                    maxlen=256)

  print("***********" + decode_review(train_data[index]))

  try:
    # Get model
    model = get_model()
    print(train_data[index])
    prediction = model.predict_classes([[train_data[index]]])

    if prediction[0][0] == 0:
      return "Bad review.."

    return "Good review!"
  except Exception as e:
    print(e)
    return e

@app.route('/predict', methods=["GET"])
def predictReview():
  review = str(request.args.get('content'))
  print("*********" + review)
  wordList = review.split(' ')

  nums = [word_index.get(i, 0) for i in wordList]

  data = tf.keras.preprocessing.sequence.pad_sequences([nums],
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)

  # Get model
  model = get_model()
  if model.predict_classes([data])[0][0] == 0:
    prediction = "Negative"
  else:
    prediction = "Positive"
    score = model.predict([data])[0][0]

  item = {}
  if prediction == "Negative":
    score = 1 - score

  item['Review'] = review
  item['Prediction'] = prediction
  item['Score'] = str(score)
  
  return jsonify(item)

@app.route('/getSamples', methods=["GET"])
def getSamples():
  start = str(request.args.get('start'))
  end = str(request.args.get('end'))
  print("*******Getting samples from " + start + " to " + end)

  result = []
  model = get_model()
  (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

  for i in range(int(start), int(end)):
    item = {}
    original = decode_review(train_data[i])
    train_data[i] = tf.keras.preprocessing.sequence.pad_sequences([train_data[i]],
                                                    value=word_index["<PAD>"],
                                                    padding='post',
                                                    maxlen=256)[0]

    score = model.predict([[train_data[i]]])[0][0]

    if model.predict_classes([[train_data[i]]])[0][0] == 0:
      prediction = "Negative"
    else:
      prediction = "Positive"

    if prediction == "Negative":
      score = 1 - score

    item['Review'] = original
    item['Prediction'] = prediction
    item['Probability'] = str(score)
    result.append(item)

  html = """\
  <table border='1'>
  <tr><th>Review</th><th>Prediction</th><th>Probability</th></tr>"""

  for row in result:
    html = html + "<tr><td>" + row['Review'] + "</td><td>" + row['Prediction'] + "</td><td>" + row['Probability'] + "</td></tr>"
  html = html + "</table>"

  return render_template(
    'result.html',
    results = html
  )
