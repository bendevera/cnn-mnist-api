import numpy as np 
import pickle
# to save algos to db after creating them
from app import db
from app.models import Algo

# for sklearn models and dataset
import sklearn 
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

algo_data = {
    'SGD': {
        'title': 'SGDClassifier',
        'description': 'Typically a binary classifier but can use OvA to apply to multiclass classificaiton problems.',
        'type': 'Gradient Descent'
    },
    'RFC': {
        'title': 'RandomForestClassifier',
        'description': 'Ensemble ML algorithm where a set of descision trees are used to "vote" on which class an image is.',
        'type': 'Ensemble'
    },
    'KN': {
        'title': 'KNeighborsClassifier',
        'description': 'Classifier implementing the k-nearest neighbors vote.',
        'type': 'Neighbors'
    },
    'SVC': {
        'title': 'LinearSVC',
        'description': 'Linear support vector classifier. Uses a hyperplane to split the dataset into the different categories.',
        'type': 'Support Vector Machine'
    },
    'CNN': {
        'title': 'ConvolutionalNeuralNetwork',
        'description': 'CNNs use a "frame"/"slide" to deduce paterns in an image or piece of data.',
        'type': 'Neural Network'
    }
}

''' MNIST MODELS:
need to create a way for user to draw a number on the react frontend 
side for these "interactable" parts of these models '''

def add_to_db(model):
    print("Adding {} to db".format(model))
    curr = Algo.query.filter_by(title=algo_data[model]['title']).first()
    if curr is None:
        curr= Algo(title=algo_data[model]['title'], 
                description=algo_data[model]['description'],
                type=algo_data[model]['type'])
        db.session.add(curr)
    else:
        curr.description = algo_data[model]['description']
        curr.type = algo_data[model]['type']
    db.session.commit()

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
    add_to_db("SGD")

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
    add_to_db("RFC")

def build_KN():
    # Builds KNeighborsClassifier
    print('Started to build K Neighbors Classifier')
    kn = KNeighborsClassifier()
    kn.fit(X_train, y_train)
    predictions = kn.predict(X_test)
    print('Accuracy score:', accuracy_score(y_test, predictions))
    path = 'app/algos/KN.pkl'
    with open(path, 'wb') as f:
        pickle.dump(kn, f)
        print('Pickled model at {}'.format(path))
    add_to_db("KN")

def buil_SVC():
    # Builds SVC
    print('Started to build SVC')
    svc = LinearSVC()
    svc.fit(X_train, y_train)
    predictions = svc.predict(X_test)
    print('Accuracy score:', accuracy_score(y_test, predictions))
    path = 'app/algos/SVC.pkl'
    with open(path, 'wb') as f:
        pickle.dump(svc, f)
        print('Pickled model at {}'.format(path))
    add_to_db("SVC")

def build_CNN():
    # Builds CNN
    print("Building CNN")
    # what is a convolutional layer? 
    # convolutional layers use a "frame"/"slide" to deduce paterns in an image or piece of data. 
    # different from recurrent layers because time or placement isn't taken into account like recurrent layers do (like for timeseries data)

    # what do the max pooling layers do? 
    # they "shrink" the size of the observation by taking the max (also can be considered the most important) value 
    # from each frame to be the "representative" of that frame for future layers to use

    # instantiating model and adding conv and pooling layers
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    # adding a classifier on top of the convnet
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    model.summary()
    model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    
    # loading and prepping images
    from keras.datasets import mnist 
    from keras.utils import to_categorical
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    train_images = train_images.reshape((60000, 28, 28, 1))
    train_images = train_images.astype('float32') / 255

    test_images = test_images.reshape((10000, 28, 28, 1))
    test_images = test_images.astype('float32') / 255 

    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

    # train_images = X_train.reshape((X_test.shape[0], 28, 28, 1))
    # train_images = train_images.astype('float32') / 255
    # test_images = X_test.reshape((10000, 28, 28, 1))
    # test_images = test_images.astype('float32') / 255 

    model.fit(train_images, train_labels, epochs=5, batch_size=64)
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print((test_loss, test_acc))
    path = 'app/algos/CNN.h5'
    model.save(path)  # creates a HDF5 file 'CNN.h5'
    print("Saved model to: {}".format(path))
    add_to_db("CNN")


if __name__ == '__main__':
    building = {
        "SGD": False, 
        "RFC": False, 
        "KN": False,
        "SVC": False,
        "CNN": True
        }
    if building["SGD"] or building["RFC"] or building["KN"]:
        print("Loading MNIST data")
        mnist = fetch_openml('mnist_784', version=1)
        X, y = mnist["data"], mnist["target"]
        y = y.astype(np.uint8)
        X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
    if building["SGD"]:
        from sklearn.linear_model import SGDClassifier
        build_SGD()
    if building["RFC"]:
        from sklearn.ensemble import RandomForestClassifier
        build_RFC()
    if building["KN"]:
        from sklearn.neighbors import KNeighborsClassifier
        build_KN()
    if building["SVC"]:
        from sklearn.svm import LinearSVC
    if building["CNN"]:
        from keras import layers
        from keras import models
        build_CNN()