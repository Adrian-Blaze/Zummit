# IMPORT ALL THE NECESSARY PACKAGES
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#LOAD DATA FROM KERAS
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
#print(len(X_test))
#plt.matshow(X_test[0])
#plt.show()

#SCALING THE DATA FOR BETTER PERFORMANCE
X_train = X_train/255
X_test = X_test/255

#X_train_flattened = X_train.reshape(len(X_train), 28*28)
#X_test_flattened = X_test.reshape(len(X_test), 28*28)
#print(X_train_flattened.shape)
#model = keras.Sequential(
#    [keras.layers.Dense(10, input_shape=(784, ), activation = 'sigmoid')]
#)
#model.compile(
#    optimizer = 'adam',
 #   loss = 'sparse_categorical_crossentropy',
  #  metrics = ['accuracy']
#)
#model.fit(X_train_flattened, y_train, epochs = 5)
#print('Evaluation...................')
#model.evaluate(X_test_flattened, y_test)
#print('With Hideen layer now.....................')
#model = keras.Sequential(
 #   [keras.layers.Dense(100, input_shape=(784, ), activation = 'relu'),
  #   keras.layers.Dense(10, activation = 'sigmoid')
   #  ]
#)
#model.compile(
 #   optimizer = 'adam',
  #  loss = 'sparse_categorical_crossentropy',
   # metrics = ['accuracy']
#)
#model.fit(X_train_flattened, y_train, epochs = 5)
#print('Without explicitly flattening.....................')

#BUILDING AND EVALUATING THE DEEP LEARNING MODEL
model = keras.Sequential([
    keras.layers.Flatten(input_shape = (28,28)),
    keras.layers.Dense(100, activation = 'relu'),
     keras.layers.Dense(10, activation = 'sigmoid')
     ])
model.compile(
    optimizer = 'adam',
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy']
)
model.fit(X_train, y_train, epochs = 5)
print(model.evaluate(X_test, y_test))