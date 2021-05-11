import pandas as pd
import tensorflow as tf
import numpy

hiddens = [2, 4, 8]
neurons = [20, 40, 80]
learning_rates = [0.02, 0.08, 0.2]

best_hidden = hiddens[0]
best_neuron = neurons[0]
best_learning_r = learning_rates[0]

df = pd.read_csv("usa_00004.csv", sep=",")

# delete first 9 columns as they are not useful to us
df = df.iloc[: , 10:]

print(df)

# remove columns 5, 7, 10 as we don't need the detailed version of these
del df['RACED']
del df['EDUCD']
del df['DEGFIELDD']

print(df)


# for h in hiddens:
#     for n in neurons:
#         for l in learning_rates:
#             model = build_model(h, n, 10, l)
