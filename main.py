import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import model_from_json
from flask import request
from flask import jsonify
from flask import Flask
from numpy import array

app = Flask(__name__)

imdb = tf.keras.datasets.imdb
# A dictionary mapping words to an integer index
word_index = imdb.get_word_index()
#graph = tf.get_default_graph()

# The first indices are reserved
word_index = {k:(v+3) for k,v in word_index.items()} 
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def get_model():
  # Model reconstruction from JSON file
  json_file = open('model_architecture.json','r')
  loaded_model_json = json_file.read()
  json_file.close()
  model = model_from_json(loaded_model_json)
  # Load weights into new model
  model.load_weights("model_weights.h5")

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
  prediction = model.predict_classes([data])

  if prediction[0][0] == 0:
    return "Bad review.."

  return "Good review!"

if __name__ == '__main__':
  app.run()
