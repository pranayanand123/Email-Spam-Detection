# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 00:27:55 2018

@author: Pranay Anand
"""

import os
import numpy as np
from collections import Counter
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.metrics import confusion_matrix 
# Create a dictionary of words with its frequency

def make_Dictionary(train_dir):
    emails = [os.path.join(train_dir,f) for f in os.listdir(train_dir)]    
    all_words = []       
    for mail in emails:    
        with open(mail) as m:
            for i,line in enumerate(m):
                if i == 2:  #Body of email is only 3rd line of text file
                    words = line.split()
                    all_words += words
    
    dictionary = Counter(all_words)
    # Paste code for non-word removal here(code snippet is given below) 
    list_to_remove = list(dictionary)
    for item in list_to_remove:
        if item.isalpha() == False: 
            del dictionary[item]
        elif len(item) == 1:
            del dictionary[item]
    dictionary = dictionary.most_common(3000)
    return dictionary

train_dir = 'train-mails'
dictionary = make_Dictionary(train_dir)

# Prepare feature vectors per training mail and its labels
def extract_features(mail_dir): 
    files = [os.path.join(mail_dir,fi) for fi in os.listdir(mail_dir)]
    features_matrix = np.zeros((len(files),3000))
    docID = 0;
    for fil in files:
      with open(fil) as fi:
        for i,line in enumerate(fi):
          if i == 2:
            words = line.split()
            for word in words:
              wordID = 0
              for i,d in enumerate(dictionary):
                if d[0] == word:
                  wordID = i
                  features_matrix[docID,wordID] = words.count(word)
        docID = docID + 1     
    return features_matrix

train_labels = np.zeros(2602)
train_labels[2170:2601] = 1
train_matrix = extract_features(train_dir)

# Training Naive bayes classifier


model = MultinomialNB()

model.fit(train_matrix,train_labels)

# Test the unseen mails for Spam
test_dir = 'test-mails'
test_matrix = extract_features(test_dir)
test_labels = np.zeros(291)
test_labels[242:291] = 1

result2 = model.predict(test_matrix)

print(confusion_matrix(test_labels,result2))
