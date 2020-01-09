from keras.models import load_model
import pickle
import numpy as np 
import os
import base64 
from io import BytesIO
from PIL import Image
import re
from app.models import Algo


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
    arr = (255 - result) / 255.
    return arr


def make_prediction(algo, params):
    if algo.title in models:
        my_model = models[algo.title]
        data = preprocess_image_2d(params['img'])
        prediction = my_model.predict_classes(data)
        return int(prediction[0])
    else:
        return "Prediction method not created for that algo yet."

# uncomment once you have run the build_algos script
# OR 
# pkl algo files is located in app/algos directory 
models = {}
models_list = Algo.query.all()
dir_path = os.path.dirname(os.path.realpath(__file__))
for model in models_list:
    path = os.path.join(dir_path, 'algos/'+model.optimizer+str(model.layers)+'.h5')
    curr_model = load_model(path)
    models[model.title] = curr_model
print(models)