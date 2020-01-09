import numpy as np 
import pickle
# to save algos to db after creating them
from app import db
from app.models import Algo
#keras imports
from keras import layers
from keras import models
#keras data imports
from keras.datasets import mnist 
from keras.utils import to_categorical

# data for db
algo_data = {
    'optimizers': ['rmsprop', 'adam'],
    'layers': [2, 3],
    'description': 'CNNs use a "frame"/"slide" to deduce paterns in an image or piece of data.',
    'type': 'Neural Network'
}

''' MNIST MODELS:
need to create a way for user to draw a number on the react frontend 
side for these "interactable" parts of these models '''

# loading and prepping images
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32') / 255 

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

def add_to_db(optimizer, layers_num):
    model = optimizer + str(layers_num)
    print("Adding {} to db".format(model))
    curr = Algo.query.filter_by(title=model).first()
    if curr is None:
        curr= Algo(title=model, 
                description=algo_data['description'],
                type=algo_data['type'],
                optimizer=optimizer,
                layers=layers_num)
        db.session.add(curr)
    else:
        curr.description = algo_data['description']
        curr.type = algo_data['type']
        curr.optimizer = optimizer
        curr.layers = layers_num
    db.session.commit()

def scale_X(X_train):
    print("scalling x data")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
    return X_train_scaled

def build_CNN(optimizer, layers_num):
    # Builds CNN
    print("Building " + optimizer+str(layers_num))
    # what is a convolutional layer? 
    # convolutional layers use a "frame"/"slide" to deduce paterns in an image or piece of data. 
    # different from recurrent layers because time or placement isn't taken into account like recurrent layers do (like for timeseries data)

    # what do the max pooling layers do? 
    # they "shrink" the size of the observation by taking the max (also can be considered the most important) value 
    # from each frame to be the "representative" of that frame for future layers to use

    # instantiating model and adding conv and pooling layers
    model = models.Sequential()
    if layers_num == 2:
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
        model.add(layers.MaxPooling2D((2,2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    elif layers_num == 3:
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
    model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    

    model.fit(train_images, train_labels, epochs=5, batch_size=64)
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print((test_loss, test_acc))
    path = 'app/algos/' + optimizer + str(layers_num) + '.h5'
    model.save(path)
    print("Saved model to: {}".format(path))
    add_to_db(optimizer, layers_num)


if __name__ == '__main__':
    # clear db 
    algos = Algo.query.all()
    for algo in algos:
        db.session.delete(algo)
    db.session.commit()

    # add all algos to db and create .h5 files
    for optimizer in algo_data['optimizers']:
        for layer in algo_data['layers']:
            build_CNN(optimizer, layer)