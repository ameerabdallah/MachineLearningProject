import pandas as pd
import tensorflow as tf
from tensorflow import keras
import numpy as np
import sklearn.model_selection as sk
import matplotlib.pyplot as plt


df = pd.read_csv("usa_00004.csv", sep=",")

# delete first 9 columns as they are not useful to us
df = df.iloc[: , 10:]
# remove columns 5, 7, 10 as we don't need the detailed version of these
del df['RACED']
del df['EDUCD']
del df['DEGFIELDD']


def build_model(n_hidden, n_neurons_hidden, n_neurons_output, learning_rate):

    #Creating the Neural Network using the Sequential API
    model = keras.models.Sequential()
    model.add(keras.layers.Flatten())                               #input layer

    #iterate over the hidden layers and create:
    for layer in range(n_hidden):
        model.add(keras.layers.Dense(n_neurons_hidden, activation="relu"))                  #hidden layer with ReLU activation function

    #output layer
    model.add(keras.layers.Dense(n_neurons_output, activation="softmax"))                #output layer with one neural for each class and the softmax activation function since the classes are exclusive

    #defining the learning rate
    opt = keras.optimizers.SGD(learning_rate)

    #Compiling the Model specifying the loss function and the optimizer to use.
    model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    return model

#classes = ['Under 15,000', '15,000 to 34,999', '35,000 to 49,999', '50,000 to 74,999', '75,000 to 99,999', '100,000 to 199,999', '200,000 and over']
features = np.array(df.values)[:,:8]
labels = np.array(df.values)[:,-1]
"""for i in range(len(labels)):
    if labels[i] < 15000:
        labels[i] = 1
    elif 15000 <= labels[i] < 35000:
        labels[i] = 2
    elif 35000 <= labels[i] < 50000:
        labels[i] = 3
    elif 50000 <= labels[i] < 75000:
        labels[i] = 4
    elif 75000 <= labels[i] < 100000:
        labels[i] = 5
    elif 100000 <= labels[i] < 199999:
        labels[i] = 6
    elif 200000 <= labels[i]:
        labels[i] = 7"""

for i in range(len(labels)):
    if labels[i] < 50000:
        labels[i] = 1
    elif 50000 <= labels[i] < 100000:
        labels[i] = 2
    elif 100000 <= labels[i] < 200000:
        labels[i] = 3
    elif 200000 <= labels[i]:
        labels[i] = 4


X_train, X_test, y_train, y_test = sk.train_test_split(features,labels,test_size=0.33, random_state = 42)

hiddens = [18, 27]
neurons = [20, 80]
learning_rates = [0.05, 0.1]

best_hidden = hiddens[0]
best_neuron = neurons[0]
best_learning_r = learning_rates[0]
highestAccuracy = 0

for h in hiddens:
    for n in neurons:
        for l in learning_rates:
            model = build_model(h, n, 10, l)

            #To train the model
            history = model.fit(X_train, y_train, epochs=5, validation_data=(X_train, y_train))
            
            #Calculate the accuracy of this neural network and store its value if it is the highest so far. To make a prediction, do:
            class_predicted = np.argmax(model.predict(X_test), axis=-1)
            score = model.evaluate(X_test, y_test, verbose=0)
            if score[1] > highestAccuracy:
                highestAccuracy = score[1]
                best_hidden, best_neuron, best_learning_r = h, n, l
            
            print("Highest SVM accuracy so far: " + str(highestAccuracy))
            print("Parameters: " + "Number of Hidden Layers: " + str(h) + ",number of neurons: " + str(n) + ",learning rate: " + str(l))
            print()

#After generating all neural networks, print the highest accuracy again and the final weights and biases of the best model
print("Highest SVM accuracy so far: " + str(highestAccuracy))

#You can generate the model again by using the hyper-parameters found before
model = build_model(best_hidden, best_neuron, 10, best_learning_r)

weights, biases = model.layers[1].get_weights()
print(weights)
print(biases)

