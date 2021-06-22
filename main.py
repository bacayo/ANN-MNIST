# Importing libraries
from sklearn.metrics import f1_score, roc_auc_score
from keras.backend import eval
from keras.layers import Dense
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import tensorflow as tf

# Importing dataset for training and test set
dataset_train = pd.read_csv('mnist_train.csv')
dataset_test = pd.read_csv('mnist_test.csv')

# Splitting data into training and test set
X_train = dataset_train.iloc[:, 1:].values
y_train = dataset_train.iloc[:, 0].values

X_test = dataset_test.iloc[:, 1:].values
y_test = dataset_test.iloc[:, 0].values

# Printing size of matrixes
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

# Feature scaling
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Building and initializing the ann
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units=128, activation='sigmoid'))

# Adding the second hidden layer
classifier.add(Dense(units=256, activation='sigmoid'))

# Adding the output layer
classifier.add(Dense(units=10, activation='sigmoid'))

# Compiling the ANN
classifier.compile(
    optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Training the ANN on the Training set
classifier.fit(X_train_scaled, y_train, batch_size=32, epochs=25)

# Calculating learning rate according to Adam algorithm
learning_rate = eval(classifier.optimizer.lr)
print(learning_rate)
classifier.summary()

y_pred = classifier.predict(X_test_scaled)

# Predicting the test set results and f1 score
test_loss = classifier.evaluate(X_test_scaled, y_test, verbose=0)
# calculate evaluation parameters

f1 = f1_score(y_test, classifier.predict_classes(
    X_test_scaled), average='micro')
stats = pd.DataFrame({'Test accuracy':  round(test_loss[1], 3),
                      'F1 score': round(f1, 3),
                      'Total Loss': round(test_loss[0], 3)}, index=[0])
# print evaluation dataframe
print(stats)
