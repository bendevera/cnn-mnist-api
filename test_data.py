import numpy as np 
from sklearn.datasets import fetch_openml

print("Loading MNIST data")
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist["data"], mnist["target"]
y = y.astype(np.uint8)
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

print(X_train.shape)
print(X_train[0].shape)
print(X_train[0])