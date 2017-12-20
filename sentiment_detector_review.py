#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 17:47:57 2017

@author: abhijeet
"""

import json
import numpy as np
import keras.preprocessing.text as kpt
from keras.preprocessing.text import Tokenizer
from keras.models import model_from_json
import pandas as pd

def convert_text_to_index_array(text):
    words = kpt.text_to_word_sequence(text)
    wordIndices = []
    for word in words:
        if word in dictionary:
            wordIndices.append(dictionary[word])
    return wordIndices

# Load the dictionary
labels = ['happy','not_happy']
with open('dictionary.json', 'r') as dictionary_file:
    dictionary = json.load(dictionary_file)

# Load trained model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights('model.h5')

testset = pd.read_csv("./test.csv")    
cLen = len(testset['Description'])
tokenizer = Tokenizer(num_words=10000)

# Predict happiness for each review in test.csv
y_pred = []   
for i in range(0,cLen):
    review = testset['Description'][i]
    testArr = convert_text_to_index_array(review)   
    input = tokenizer.sequences_to_matrix([testArr], mode='binary')
    pred = model.predict(input)
    #print pred[0][np.argmax(pred)] * 100, labels[np.argmax(pred)]
    y_pred.append(labels[np.argmax(pred)])


# Write the results in submission csv file
raw_data = {'User_ID': testset['User_ID'], 
        'Is_Response': y_pred}
df = pd.DataFrame(raw_data, columns = ['User_ID', 'Is_Response'])
df.to_csv('submission_model1.csv', sep=',',index=False)