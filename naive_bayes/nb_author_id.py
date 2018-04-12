#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess

# Complete naive bayes imports
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

gnb = GaussianNB()
t0 = time()
gnb.fit(features_train, labels_train)
print "Time for training", round(time()-t0, 3), "s"

t1 = time()
prediction = gnb.predict(features_test)
print "Time for prediction", round(time()-t1, 3), "s"

print "Accuracy for Naive Bayes is ", accuracy_score(labels_test, prediction)

#########################################################
### your code goes here ###


#########################################################


