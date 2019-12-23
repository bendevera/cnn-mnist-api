from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
import pickle
import numpy as np 
import os

def make_prediction(algo, params):
    if algo.title == 'SGDClassifier':
        # could have data be an index of the test data
        data = np.array(params['data'])
        # data = scaler.fit_transform(data.reshape(28, 28))
        prediction = sgd_clf.predict([data])
        return prediction 
    else:
        return "Prediction method not created for that algo yet."

sgd_clf = SGDClassifier()
dir_path = os.path.dirname(os.path.realpath(__file__))
sgd_path = os.path.join(dir_path, 'lib/algos/SGDClassifier.pkl')
with open(sgd_path, 'rb') as f:
    sgd_clf = pickle.load(f)

scaler = StandardScaler()