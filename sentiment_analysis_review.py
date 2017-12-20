#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 16:51:38 2017

@author: abhijeet
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import json
import keras
import keras.preprocessing.text as kpt
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense, Dropout

def convert_text_to_index_array(text):
    return [dictionary[word] for word in kpt.text_to_word_sequence(text)]

def data_prepare(training_file_path):

    dataset = pd.read_csv(training_file_path)
    reviews = []
    labels = []    
    
    # Enconding Categorical Data     
    labelencoder_y = LabelEncoder()
    dataset['Is_Response'] = labelencoder_y.fit_transform(dataset['Is_Response'])
    cLen = len(dataset['Description'])
        
    for i in range(0,cLen):
        review = dataset['Description'][i]
        reviews.append(review) 
        label = dataset["Is_Response"][i]
        labels.append(label)    
    labels = np.asarray(labels)
    return reviews,labels


train_file_path = "./train.csv"
[reviews,labels] = data_prepare(train_file_path)

# Create Dictionary of words and their indices
max_words = 10000
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(reviews)
dictionary = tokenizer.word_index

# save dictionary
with open('dictionary.json','w') as dictionary_file:
    json.dump(dictionary,dictionary_file)

# Replace words of each text review to indices
allWordIndices = []
for num,text in enumerate(reviews):
    wordIndices = convert_text_to_index_array(text)
    allWordIndices.append(wordIndices)

# Convert the index sequences into binary bag of words vector (one hot encoding) 
allWordIndices = np.asarray(allWordIndices)
train_X = tokenizer.sequences_to_matrix(allWordIndices, mode='binary')
labels = keras.utils.to_categorical(labels,num_classes=2)

# Creating Dense Neural Networl Model
model = Sequential()
model.add(Dense(256, input_shape=(max_words,), activation='elu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='elu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

print model.summary()

model.compile(loss='categorical_crossentropy',
  optimizer='sgd',
  metrics=['accuracy'])

# Training the Model
model.fit(train_X, labels,
  batch_size=32,
  epochs=10,
  verbose=1,
  validation_split=0.1,
  shuffle=True)


# Save model to disk
model_json = model.to_json()
with open('model.json', 'w') as json_file:
    json_file.write(model_json)

model.save_weights('model.h5')    
