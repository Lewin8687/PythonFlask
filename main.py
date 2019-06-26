import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import model_from_json
from flask import request
from flask import jsonify
from flask import Flask
from numpy import array

global graph, model

app = Flask(__name__)

def get_model():
  # Model reconstruction from JSON file
  with open('model_architecture.json', 'r') as f:
      model = model_from_json(f.read())

  # Load weights into the new model
  model.load_weights('model_weights.h5')

  print("Model loaded!")

@app.route('/')
def hello_world():
  return 'Hey its Python Flask application!'

print("Loading model...")
get_model()
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

@app.route('/predict', methods=["GET"])
def predict():
  (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

  print(train_data[0])

  # Preprocess review
  train_data = tf.keras.preprocessing.sequence.pad_sequences(train_data,
                                                    value=word_index["<PAD>"],
                                                    padding='post',
                                                    maxlen=256)

  print("******************1")

  try:
    print(model)
    prediction = model.predict([train_data[0]])

    print("**************************2")
    print(prediction)

    return prediction
  except Exception as e:
    print(e)
    return e

if __name__ == '__main__':
  app.run()
