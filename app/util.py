from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
import pickle
import numpy as np 
import os
import base64 
from io import BytesIO
from PIL import Image

def preprocess_image(img):
    starter = img.find(',')
    image_data = img[starter+1:]
    image_data = bytes(image_data, encoding='ascii')
    im = Image.open(BytesIO(base64.b64decode(image_data)))
    im = im.convert('1')
    im.thumbnail((28, 28), Image.ANTIALIAS)
    im = np.array(im)
    im = im.reshape(1, -1)
    result = im.copy()
    return result

def make_prediction(algo, params):
    # going to add preprocessing for data here
    data = preprocess_image(params['img'])
    if algo.title == 'SGDClassifier':
        # could have data be an index of the test data
        data = scaler.fit_transform(data)
        prediction = sgd_clf.predict(data)
        return prediction[0]
    else:
        return "Prediction method not created for that algo yet."

sgd_clf = SGDClassifier()
dir_path = os.path.dirname(os.path.realpath(__file__))
sgd_path = os.path.join(dir_path, 'lib/algos/SGDClassifier.pkl')
with open(sgd_path, 'rb') as f:
    sgd_clf = pickle.load(f)

scaler = StandardScaler()