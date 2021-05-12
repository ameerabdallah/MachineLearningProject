import pandas as pd
import tensorflow as tf
from tensorflow import keras
import numpy
import math
import sklearn.model_selection as sk
from tensorflow.python.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

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

def delete_extra_fields(df):
    del df['RACED']
    del df['EDUCD']
    del df['DEGFIELDD']
    del df['EMPSTATD']
    del df['CLASSWKRD']
    del df['IND']
    del df['SCHLTYPE']
    return df

def preprocess_dataset(x, y, classes):

    class_bounds = [48500, 150000]
    
    for i, label in enumerate(y):
        if class_bounds[0] > label:
            y[i] = 0
            continue
        over = True
        for j, bound in enumerate(class_bounds[1:]):
            if class_bounds[j] <= label < bound:
                y[i] = j+1
                over = False
                break
        if over:
            y[i] = len(classes)-1
            

    return x, y
    # classify 



def create_dictionary_from_cbk(file_name):

    columns_to_remove=[
        'RACED',
        'EDUCD',
        'DEGFIELDD',
        'EMPSTATD',
        'CLASSWKRD',
        'IND',
        'SCHLTYPE',
    ]

    with open(file_name) as file:
        features = file.read().split("\n\n")
        feature_names=[]
        my_dict = {
            '':{0:''}
            }
        for feature in features:
            labels = feature.splitlines()
            if len(labels) == 0:
                break
            title=labels[0].split()[0]
            if title in columns_to_remove:
                continue
            feature_names.append(title)
            my_dict[title] = {int(labels[1].split()[0]):labels[1].split()[1]}
            for line in labels[2:]:
                key = line.split()[0]
                value = line.split()[1]
                my_dict[title][int(key)] = value
                
        return my_dict, feature_names
        

dict_cbk, feature_names = create_dictionary_from_cbk('usa_00005.cbk')

# print(dict_cbk['AGE'][0])
# print(feature_names)

df = pd.read_csv("usa_00005.csv", sep=",")

# delete first 10 columns as they are not useful to us
df = df.iloc[: , 10:]

# remove columns 5, 7, 10 as we don't need the detailed version of these
df = delete_extra_fields(df)

#seperate dataframe into features and labels (dfx and dfy)
dfx = df.loc[:, df.columns != 'INCTOT']
dfy = df.loc[:,'INCTOT']

del df

classes = ['Under $48,500 (Lower-Income)', '$48,500 to $149,99 (Middle-Income)', '$150,000 and over (Upper-Income)']

x = numpy.array(dfx.values)
y = numpy.array(dfy.values)

del dfx
del dfy

x, y = preprocess_dataset(x, y, classes)

X_train, X_test, y_train, y_test = sk.train_test_split(x,y,test_size=0.33, random_state = 15)


hiddens = [6, 8, 10]
neurons = [20, 50, 70]
learning_rates = [0.01, 0.05, 0.1]

best_hidden = hiddens[0]
best_neuron = neurons[0]
best_learning_r = learning_rates[0]
highestAccuracy = 0

early_stopping = EarlyStopping(monitor ="val_loss", 
                                mode ="min", patience = 5, 
                                restore_best_weights = True)

best_model = build_model(best_hidden, best_neuron, len(classes), best_learning_r)

for h in hiddens:
    for n in neurons:
        for l in learning_rates:
            model = build_model(h, n, len(classes), l)

            #To train the model
            history = model.fit(X_train, y_train, batch_size=1024, epochs=1000, validation_data=(X_test, y_test), callbacks=[early_stopping])
            
            #Calculate the accuracy of this neural network and store its value if it is the highest so far. To make a prediction, do:
            class_predicted = numpy.argmax(model.predict(X_test), axis=-1)
            print(class_predicted[0])
            score = model.evaluate(X_test, y_test, verbose=1)

            if score[1] > highestAccuracy:
                highestAccuracy = score[1]
                best_hidden, best_neuron, best_learning_r= h, n, l
                best_model = model
            
            print("Highest SVM accuracy so far: " + str(highestAccuracy))
            print("Parameters: " + "Number of Hidden Layers: " + str(h) + ", number of neurons: " + str(n) + ",learning rate: " + str(l))
            print()
            

print("Best Parameters: " + "Number of Hidden Layers: " + str(best_hidden) + ",number of neurons: " + str(best_neuron) + ",learning rate: " + str(best_learning_r) )


# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()