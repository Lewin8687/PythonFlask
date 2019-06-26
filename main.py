import keras
from keras import backend as K
from keras.models import Sequential
from keras.models import load_model
from flask import request
from flask import jsonify
from flask import Flask

app = Flask(__name__)

def get_model():
  global model
  model = load_model('FirstTrainedModel.h5')
  print("Model loaded!")

@app.route('/')
def hello_world():
  return 'Hey its Python Flask application!'

@app.route('/predict', methods=['POST'])
def predict():
  content = request.get_json(force=True)
  review = content['review']
  
  prediction = model.predict(review).tolist()
  
  response = {
    'prediction':{
      'result': prediction[0]
    }
  }

if __name__ == '__main__':
  app.run()
