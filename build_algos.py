import sklearn 
import numpy as np 
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
import pickle
# to save algos to db after creating them
# from app import db
# from app.models import Algo

algo_data = {
    'SGD': {
        'title': 'SGDClassifier',
        'description': 'Typically a binary classifier but can use OvA to apply to multiclass classificaiton problems.',
        'supervized': True,
        'regression': False,
        'type': ''
    }
}

''' MNIST MODELS:
need to create a way for user to draw a number on the react frontend 
side for these "interactable" parts of these models '''
def build_SGDClassifier():
    # Builds Stoca
    print('Started to build SGD Classifier')
    mnist = fetch_openml('mnist_784', version=1)
    X, y = mnist["data"], mnist["target"]
    y = y.astype(np.uint8)
    X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
    sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)
    sgd_clf.fit(X_train_scaled, y_train)
    predictions = sgd_clf.predict(scaler.fit_transform(X_test.astype(np.float64)))
    print('Accuracy score:', accuracy_score(y_test, predictions))
    path = 'app/lib/algos/SGDClassifier.pkl'
    with open(path, 'wb') as f:
        pickle.dump(sgd_clf, f)
        print('Pickeled model at {}'.format(path))
    SGD = Algo(title=algo_data['SGD']['title'], 
               description=algo_data['SGD']['description'],
               supervised=algo_data['SGD']['supervised'],
               regression=algo_data['SGD']['regression'],
               type=algo_data['SGD']['type'])
    db.session.add(SGD)
    db.session.commit()
    

if __name__ == '__main__':
    build_SGDClassifier()