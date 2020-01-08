from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from keras.models import load_model

import pickle
import numpy as np 
import os
import base64 
from io import BytesIO
from PIL import Image
import re

def preprocess_image_scaled(img):
    starter = img.find(',')
    image_data = img[starter+1:]
    image_data = bytes(image_data, encoding='ascii')
    im = Image.open(BytesIO(base64.b64decode(image_data)))
    im = im.convert('L')
    im.thumbnail((28, 28), Image.ANTIALIAS)
    # im.show()
    im = np.array(im) # shape: (28, 28)
    im = im.reshape(1, -1) # convert to (1, 784)
    result = im.copy()
    # Normalize and invert pixel values
    # result = (255 - result) / 255.
    result = scaler.fit_transform(result)
    return result

def preprocess_image_standard(img):
    starter = img.find(',')
    image_data = img[starter+1:]
    image_data = bytes(image_data, encoding='ascii')
    im = Image.open(BytesIO(base64.b64decode(image_data)))
    im = im.convert('L')
    im.thumbnail((28, 28), Image.ANTIALIAS)
    # im.show()
    im = np.array(im) # shape: (28, 28)
    im = im.reshape(1, -1) # convert to (1, 784)
    result = im.copy()
    return result

def preprocess_image_2d(img):
    starter = img.find(',')
    image_data = img[starter+1:]
    image_data = bytes(image_data, encoding='ascii')
    im = Image.open(BytesIO(base64.b64decode(image_data)))
    im = im.convert('L')
    im.thumbnail((28, 28), Image.ANTIALIAS)
    # im.show()
    im = np.array(im) # shape: (28, 28)
    result = im.copy().reshape(1, 28, 28, 1)
    # result = (255 - result) / 255.
    # return result
    # Convert data url to numpy array
    # img_str = re.search(r'base64,(.*)', img).group(1)
    # image_bytes = BytesIO(base64.b64decode(img_str))
    # im = Image.open(image_bytes)
    # arr = np.array(im)[:,:,0:1]

    # Normalize and invert pixel values
    arr = (255 - result) / 255.
    print(arr.shape)
    return arr
# Convert data url to numpy array
# img_str = re.search(r'base64,(.*)', img).group(1)
# image_bytes = BytesIO(base64.b64decode(img_str))
# im = Image.open(image_bytes)
# arr = np.array(im)[:,:,0:1]

# # Normalize and invert pixel values
# arr = (255 - arr) / 255.
# print(arr.shape)
# return arr

def make_prediction(algo, params):
    if algo.title == 'SGDClassifier':
        data = preprocess_image_scaled(params['img'])
        prediction = sgd_clf.predict(data)
        return int(prediction[0])
    elif algo.title == 'RandomForestClassifier':
        data = preprocess_image_standard(params['img'])
        prediction = RFC.predict(data)
        return int(prediction[0])
    elif algo.title == 'KNeighborsClassifier':
        data = preprocess_image_standard(params['img'])
        prediction = KN.predict(data)
        return int(prediction[0])
    elif algo.title == 'ConvolutionalNeuralNetwork':
        data = preprocess_image_2d(params['img'])
        print(data)
        prediction = CNN.predict_classes(data)
        print(prediction)
        return int(prediction[0])
    else:
        return "Prediction method not created for that algo yet."

# uncomment once you have run the build_algos script
# OR 
# pkl algo files is located in app/algos directory 

sgd_clf = SGDClassifier()
dir_path = os.path.dirname(os.path.realpath(__file__))
sgd_path = os.path.join(dir_path, 'algos/SGDClassifier.pkl')
with open(sgd_path, 'rb') as f:
    sgd_clf = pickle.load(f)

RFC = RandomForestClassifier()
rfc_path = os.path.join(dir_path, 'algos/RFC.pkl')
with open(rfc_path, 'rb') as f:
    RFC = pickle.load(f)

KN = KNeighborsClassifier()
kn_path = os.path.join(dir_path, 'algos/KN.pkl')
with open(kn_path, 'rb') as f:
    KN = pickle.load(f)

CNN_path = os.path.join(dir_path, 'algos/CNN.h5')
CNN = load_model(CNN_path)

scaler = StandardScaler()