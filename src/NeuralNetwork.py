import numpy as np
import pandas as pd
import helpers
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict, GridSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, precision_recall_curve, roc_curve
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

#load data
X_raw_data = pd.read_csv('../data/multiclass/X.csv', header=None)
y_raw_data = pd.read_csv('../data/multiclass/y.csv', header=None)

#remove the training set
X_training, X_testing, y_training, y_testing = train_test_split(X_raw_data, y_raw_data, test_size = 0.2, random_state = 78, stratify=y_raw_data)

#initialize the model
mlp = MLPClassifier(hidden_layer_sizes=(500,), solver='sgd')

#standardize the data
scaler = preprocessing.StandardScaler().fit(X_training)

#create pipeline & grid
pipeline = Pipeline([('scaler', scaler),
        ('model', mlp)])

grid = [{'model__max_iter': [1000]}]

#Perform Grid Search and Cross Validation
clf = GridSearchCV(pipeline, param_grid = grid, cv=5, refit = True)

#train the model
clf.fit(X_training, y_training)

#cross validation - use cross_val_predict to give the actual values
y_train_prediction = cross_val_predict(clf, X_testing, y_testing, cv=5)

#confusion matrix and visualization
confusion_matrix = confusion_matrix(y_testing, y_train_prediction)
print(confusion_matrix)
xlabels=["air", "book", "hand", "knife", "plastic case"]
ylabels=["air", "book", "hand", "knife", "plastic case"]
title = "Neural Network Confusion Matrix"
helpers.confusionMatrix(confusion_matrix, xlabels, ylabels, title)

#performance evaluation of training data - overall
#micro is better if there is a class imbalance
print("Precision is: ")                                 #True Positive / (True Positive + False Positive)
print(precision_score(y_testing, y_train_prediction, average='micro'))
print("Recall is: ")                                    #True Positive / (True Positive + False Negative)
print(recall_score(y_testing, y_train_prediction, average='micro'))
print("F1 Score is: ")                                  #useful for comparing two classifiers
print(f1_score(y_testing, y_train_prediction, average='micro'))
