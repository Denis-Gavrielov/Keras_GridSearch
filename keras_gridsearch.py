
'''
This is a project where I am creating a grid search cross validation 
with Keras and scikit-learn for deep learning. My aim is to automate 
the hyperparameter tuning and maximise the accuracy with a script that 
could be applicable to multiple problems with only little adjustments.

I used the pre-processing ideas from Udacity and got also inspired by 
this post: 
https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/

I am very happy to take constructive criticism, advice, and praise. :) 
'''

# Imports
import numpy as np
import keras
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.preprocessing.text import Tokenizer
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV
#from sklearn.model_selection import GridSearchCV
## I am using RandomizedSearchCV. If I had endless time and computational
## power I would happily use GridSearchCV instead
from keras.wrappers.scikit_learn import KerasClassifier
from keras.constraints import maxnorm

#np.random.seed(42)


# ## 1. Loading the data
# This dataset comes preloaded with Keras, so one simple command will 
# get us training and testing data. There is a parameter for how many 
# words we want to look at. We've set it at 1000, but feel free to experiment.

# Loading the data (it's preloaded in Keras)
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=1000)

# print(x_train.shape)
# print(x_test.shape)
## when working with jupyter notebook we can take this step to explore the data

# ## 2. Examining the data
# Notice that the data has been already pre-processed, where all the words 
# have numbers, and the reviews come in as a vector with the words that the 
# review contains. For example, if the word 'the' is the first one in our 
# dictionary, and a review contains the word 'the', then there is a 1 in 
# the corresponding vector.
# 
# The output comes as a vector of 1's and 0's, where 1 is a positive 
# sentiment for the review, and 0 is negative.


# print(x_train[0])
# print(y_train[0])
## again, for jupyter notebooks


# ## 3. One-hot encoding the output
# Here, we'll turn the input vectors into (0,1)-vectors. For example, if the 
# pre-processed vector contains the number 14, then in the processed vector, 
# the 14th entry will be 1.


# One-hot encoding the output into vector mode, each of length 1000
tokenizer = Tokenizer(num_words=1000)
x_train = tokenizer.sequences_to_matrix(x_train, mode='binary')
x_test = tokenizer.sequences_to_matrix(x_test, mode='binary')
# print(x_train[0])


# And we'll also one-hot encode the output.

# One-hot encoding the output
num_classes = 2
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
# print(y_train.shape)
# print(y_test.shape)

# print(x_train.shape)


## up until here I got the code from Udacity. Below is the meat of my project:


## I created a function to create the model with different hyperparameters.
## All relevant hyperparameters can be seen in the input of the funcion, 
## except for the batch_size and epochs. 

def create_model(neurons=2, 
                 dropout_rate=0.0, 
                activation='relu', 
                 init_mode='uniform', 
                 optimizer='adam',
                 hidden_layers=0):

    #create model
    model = Sequential()
    model.add(Dense(neurons, input_dim=1000, 
                    kernel_initializer=init_mode, activation=activation))
    ## as mentioned above we have 1000 dummy variables representing 1000 words,
    ## hence we need the input_dim=1000. 
    model.add(Dropout(dropout_rate))

    ## here I am making it possible to fine tune the amount of layers 
    ## in our perceptron. These layers will have the same attributes,
    ## as our first layer (except of the missing "input_dim")
    if hidden_layers == 0:
        pass
    else:
        for i in range(hidden_layers):
            model.add(Dense(neurons, kernel_initializer=init_mode, 
                activation=activation))
            model.add(Dropout(dropout_rate))


    model.add(Dense(2, kernel_initializer='uniform', activation='sigmoid'))
    #compile model
    model.compile(loss='binary_crossentropy', optimizer=optimizer,
                 metrics=['accuracy'])
    return model

# fix random seed for reproductibility
# seed =7 
# np.random.seed(seed)

#create model
model = KerasClassifier(build_fn=create_model, verbose=2)
#define the grid search parameters

batch_size = [8, 16, 32] 
epochs = [10, 25] ## increasing the number of epochs can make the grid search 
## significantly more time consuming. 
neurons = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024] 
dropout_rate= [0.0, 0.2, 0.4, 0.6, 0.8]
hidden_layers = [0, 1, 2, 3, 4, 5] ## I have tried large numbers like 64...
## the score was horrible, probably due to overfitting. 

# activation = ['softmax', 'relu', 'tanh', 'linear'] 
## I could also only try these 4 activation functions
activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 
               'hard_sigmoid', 'linear']
init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero', 
             'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
# optimizer = ['adam', 'adagrad', 'adamax', 'rmsprop']
## I could also only try these 4 optimizers 
optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']

param_grid = dict(batch_size=batch_size,
    epochs=epochs,
    hidden_layers=hidden_layers,
    neurons=neurons, dropout_rate=dropout_rate,
                  activation=activation,
                  init_mode=init_mode, 
                  optimizer=optimizer)
grid = RandomizedSearchCV(estimator=model, param_distributions=param_grid,
                         n_iter=10, n_jobs=-1, cv=4, verbose=10)
## feel free to change n_iter. It is the number of grid combinations that 
## are being tried out. The more the better, but also the more time consuming.
## n_jobs=-1 to use all available cores
grid_result = grid.fit(x_train, y_train)

## print the results and also save them in a .txt file.
f = open('results.txt', 'w')

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
f.write("Best: %f using %s\n\n" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params'] 
used = []

for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
    f.write("%f (%f) with: %r\n\n" % (mean, stdev, param))
    used.append(param)


########################################
## train the actual model with the best parameters 

best = grid_result.best_params_

model = Sequential()
model.add(Dense(best['neurons'], input_dim=1000, activation=best['activation']))
model.add(Dropout(best['dropout_rate']))
if best['hidden_layers'] == 0:
    pass
else:
    for i in range(best['hidden_layers']):
            model.add(Dense(best['neurons'], 
                activation=best['activation']))
            model.add(Dropout(best['dropout_rate']))
model.add(Dense(2, kernel_initializer='uniform', activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer=best['optimizer'], metrics=['accuracy'])

hist = model.fit(x_train, y_train, 
                batch_size=best['batch_size'],
                epochs=best['epochs'],
                validation_data=(x_test, y_test),
                verbose=2)

## print the final results from unseen data
score = model.evaluate(x_test, y_test, verbose=0)
print("Accuracy: ", score[1])
f.write(20*'-' + '\nAccuracy of best model on unseen data: ' + str(score[1]) + \
    '\n\nParameters used: ' + str(best))
f.close()
import pprint
print('using:')
pprint.pprint(best)
print('COMBINATIONS TRIED')
pprint.pprint(used)

"""
So far I am happy with my result. I have always achieved an accuracy score 
of over 85% with this script. My next addition to this project would be 
having a second stage grid search, where I use the best parameters from 
the first grid search, to make a second grid search tuning hyperparameters 
close to my first results and hence maximising accuracy even more.

Also, I want to try to implement a timer to see how long each model takes.
""" 