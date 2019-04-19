import numpy as np
import pandas as pd
import helpers
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict, GridSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, precision_recall_curve, roc_curve
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

#load data
X_raw_data = pd.read_csv('../data/binary/X.csv', header=None)
y_raw_data = pd.read_csv('../data/binary/y.csv', header=None)

X_means = X_raw_data.iloc[:,:256] # this takes only the means

#explore the data
#helpers.checkDataForNullAndType(X_raw_data, y_raw_data)

#remove the training set from whole data set
X_training, X_testing, y_training, y_testing = train_test_split(X_raw_data, y_raw_data, test_size = 0.2, random_state = 78, stratify=y_raw_data)

#create a training and testing dataset from the mean values
X_training_means, X_testing_means, y_training_means, y_testing_means = train_test_split(X_means, y_raw_data, test_size = 0.2, random_state = 78, stratify=y_raw_data)

#visualize the data
helpers.visualizeOneRowOfData(X_training)
helpers.visualizeOneRowOfData(X_training_means)
helpers.visualizeStandardDeviation(X_training)
helpers.visualizeAllRowsOfData(X_training)
helpers.visualizeAllRowsOfData(X_training_means)

#heatmap
helpers.correlationMatrix(X_training, "Heatmap of X Training Data")
helpers.correlationMatrix(X_training_means, "Heatmap of X Means Training Data")

#initialize the model - stochasic gradient descent classifier
sgd_clf = SGDClassifier(random_state=45, max_iter=1000, tol=1e-3)

#standardize the data
scaler = StandardScaler()
scaler.fit(X_training)
x_train = scaler.transform(X_training)
X_testing = scaler.transform(X_testing)

#train the model
sgd_clf.fit(x_train, y_training)

#cross validation - use cross_val_predict to give the actual values
y_train_prediction = cross_val_predict(sgd_clf, X_testing, y_testing, cv=5)

#caclulate the score for each training instance, then use it to plot Precision-Recall Curve and Receiver Operating Characteristic
y_scores = cross_val_predict(sgd_clf, X_testing, y_testing, cv=5, method="decision_function")

#performance evaluation
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
title = "SGD Binary Classification Confusion Matrix"
helpers.confusionMatrix(confusion_matrix, xlabels, ylabels, title)

#results visualization - Precision-Recall Curve - training data
precisions, recalls, thresholds = precision_recall_curve(y_testing, y_scores)
helpers.plot_precision_recall_curve(precisions, recalls)
helpers.plot_precision_recall_vs_threshold(precisions, recalls, thresholds)

#results visualization - Receiver Operating Characteristic - training data
fpr, tpr, thresholds = roc_curve(y_testing, y_scores)
helpers.plot_roc_curve(fpr, tpr)
