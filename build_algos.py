import sklearn 
import numpy as np 
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
# to save algos to db after creating them
from app import db
from app.models import Algo

algo_data = {
    'SGD': {
        'title': 'SGDClassifier',
        'description': 'Typically a binary classifier but can use OvA to apply to multiclass classificaiton problems.',
        'type': 'Gradient Descent'
    },
    'RFC': {
        'title': 'RandomForestClassifier',
        'description': 'Ensemble ML algorithm where a set of descision trees are used to "vote" on which class an image is',
        'type': 'Ensemble'
    }

}

''' MNIST MODELS:
need to create a way for user to draw a number on the react frontend 
side for these "interactable" parts of these models '''

print("Loading MNIST data")
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist["data"], mnist["target"]
y = y.astype(np.uint8)
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

def scale_X(X_train):
    print("scalling x data")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
    return X_train_scaled

def build_SGD():
    # Builds Stochastic Gradient Descent Classifier
    print('Started to build SGD Classifier')
    X_train_scaled = scale_X(X_train)
    sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)
    sgd_clf.fit(X_train_scaled, y_train)
    predictions = sgd_clf.predict(scale_X(X_test))
    print('Accuracy score:', accuracy_score(y_test, predictions))
    path = 'app/algos/SGDClassifier.pkl'
    with open(path, 'wb') as f:
        pickle.dump(sgd_clf, f)
        print('Pickeled model at {}'.format(path))
    SGD = Algo.query.filter_by(title=algo_data['SGD']['title']).first()
    if SGD is None:
        SGD = Algo(title=algo_data['SGD']['title'], 
                description=algo_data['SGD']['description'],
                type=algo_data['SGD']['type'])
        db.session.add(SGD)
    else:
        SGD.description = algo_data['SGD']['description']
        SGD.type = algo_data['SGD']['type']
    db.session.commit()

def build_RFC():
    # Builds Random Forest Classifier 
    print('Started to build Random Forest Classifier')
    rfc = RandomForestClassifier()
    rfc.fit(X_train, y_train)
    predictions = rfc.predict(X_test)
    print('Accuracy score:', accuracy_score(y_test, predictions))
    path = 'app/algos/RFC.pkl'
    with open(path, 'wb') as f:
        pickle.dump(rfc, f)
        print('Pickled model at {}'.format(path))
    RFC = Algo.query.filter_by(title=algo_data['RFC']['title']).first()
    if RFC is None:
        RFC = Algo(title=algo_data['RFC']['title'], 
                description=algo_data['RFC']['description'],
                type=algo_data['RFC']['type'])
        db.session.add(RFC)
    else:
        RFC.description = algo_data['RFC']['description']
        RFC.type = algo_data['RFC']['type']
    db.session.commit()
    

if __name__ == '__main__':
    build_SGD()
    build_RFC()