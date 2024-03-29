import numpy as np
import pandas as pd
import helpers
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict, GridSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, precision_recall_curve, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


#load data
X_raw_data = pd.read_csv('../data/binary/X.csv', header=None)
y_raw_data = pd.read_csv('../data/binary/y.csv', header=None)

X_mean = X_raw_data.loc[:,:255] # this takes only the means

#explore the data
helpers.checkDataForNullAndType(X_raw_data, y_raw_data)

#remove the training set
X_training, X_testing, y_training, y_testing = train_test_split(X_raw_data, y_raw_data, test_size = 0.2, random_state = 78, stratify=y_raw_data)

#visualize the data
helpers.visualizeOneRowOfData(X_training)
helpers.visualizeOneRowOfData(X_mean)
helpers.visualizeStandardDeviation(X_training)
helpers.visualizeAllRowsOfData(X_training)
helpers.histogram(y_training, 3, 'Histogram of Y Values For Binary Classification, With 2 Bins')

#initialize the model
log_reg = LogisticRegression(solver='liblinear') #good choice for small datasets

#standardize the data
scaler = StandardScaler()
scaler.fit(X_training)
x_training = scaler.transform(X_training)
X_testing = scaler.transform(X_testing)

#train the model
log_reg.fit(X_training, y_training)

#cross validation - use cross_val_predict to give the actual values
y_train_prediction = cross_val_predict(log_reg, X_testing, y_testing, cv=5)

#caclulate the score for each training instance, then use it to plot Precision-Recall Curve and Receiver Operating Characteristic
y_scores = cross_val_predict(log_reg, X_testing, y_testing, cv=5, method="decision_function")

#performance evaluation of training data
confusion_matrix = confusion_matrix(y_testing, y_train_prediction)
print(confusion_matrix)
print("Precision is: ")                                 #True Positive / (True Positive + False Positive)
print(precision_score(y_testing, y_train_prediction))
print("Recall is: ")                                    #True Positive / (True Positive + False Negative)
print(recall_score(y_testing, y_train_prediction))
print("F1 Score is: ")                                  #useful for comparing two classifiers
print(f1_score(y_testing, y_train_prediction))

#visualize confusion_matrix
xlabels=["book", "plastic case"]
ylabels=["book", "plastic case"]
title = "Logistic Regression Confusion Matrix"
helpers.confusionMatrix(confusion_matrix, xlabels, ylabels, title)

#results visualization - Precision-Recall Curve - training data
precisions, recalls, thresholds = precision_recall_curve(y_testing, y_scores)
helpers.plot_precision_recall_curve(precisions, recalls)
helpers.plot_precision_recall_vs_threshold(precisions, recalls, thresholds)

#results visualization - Receiver Operating Characteristic - training data
fpr, tpr, thresholds = roc_curve(y_testing, y_scores)
helpers.plot_roc_curve(fpr, tpr)
