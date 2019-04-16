import numpy as np
import pandas as pd
import helpers
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict, GridSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, precision_recall_curve, roc_curve
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline


#load data
X_raw_data = pd.read_csv('../data/multiclass/X.csv', header=None)
y_raw_data = pd.read_csv('../data/multiclass/y.csv', header=None)

X_means = X_raw_data.loc[:,:255] # this takes only the means

#explore the data
helpers.checkDataForNullAndType(X_raw_data, y_raw_data)

#remove the training set
X_training, X_testing, y_training, y_testing = train_test_split(X_raw_data, y_raw_data, test_size = 0.2, random_state = 78, stratify=y_raw_data)

#create a training and testing dataset from the mean values
X_training_means, X_testing_means, y_training_means, y_testing_means = train_test_split(X_means, y_raw_data, test_size = 0.2, random_state = 78, stratify=y_raw_data)

#visualize the data
helpers.visualizeOneRowOfData(X_training)
helpers.visualizeOneRowOfData(X_training_means)
helpers.visualizeStandardDeviation(X_training)
#helpers.visualizeAllRowsOfData(X_training)
#helpers.visualizeAllRowsOfData(X_training_means)

#heatmap
helpers.correlationMatrix(X_means)

#initialize the model
log_reg = LogisticRegression(penalty='l2', multi_class='multinomial', solver='newton-cg')
#multinomial performs multiclass. Can use one of three solvers, all of which
#require 'l2' regularization, default is newton-cg

#standardize the data
scaler = preprocessing.StandardScaler().fit(X_training)

#create pipeline & grid
pipeline = Pipeline([('scaler', scaler),
        ('model', log_reg)])

grid = [{'model__tol': [1e-3, 1e-4, 1e-5],
        'model__max_iter': [1000]}]

#Perform Grid Search and Cross Validation
clf = GridSearchCV(pipeline, param_grid = grid, cv=5, refit = True)

#train the model
clf.fit(X_training, y_training)

#cross validation - use cross_val_predict to give the actual values
y_train_prediction = cross_val_predict(clf, X_training, y_training, cv=5)

#calculate the score for each training instance, then use it to plot Precision-Recall Curve and Receiver Operating Characteristic
y_scores = cross_val_predict(clf, X_training, y_training, cv=5, method="decision_function")

#performance evaluation of training data - per class
print(confusion_matrix(y_training, y_train_prediction))
print("Precision is: ")                                 #True Positive / (True Positive + False Positive)
print(precision_score(y_training, y_train_prediction, average=None))
print("Recall is: ")                                    #True Positive / (True Positive + False Negative)
print(recall_score(y_training, y_train_prediction, average=None))
print("F1 Score is: ")                                  #useful for comparing two classifiers
print(f1_score(y_training, y_train_prediction, average=None))

#performance evaluation of training data - overall
print(confusion_matrix(y_training, y_train_prediction))
print("Precision is: ")                                 #True Positive / (True Positive + False Positive)
print(precision_score(y_training, y_train_prediction, average='micro'))
print("Recall is: ")                                    #True Positive / (True Positive + False Negative)
print(recall_score(y_training, y_train_prediction, average='micro'))
print("F1 Score is: ")                                  #useful for comparing two classifiers
print(f1_score(y_training, y_train_prediction, average='micro'))

#results visualization - Precision-Recall Curve - training data
#need one hot encoding for these to work, multiclass is not supported --> label_binarize
#precisions, recalls, thresholds = precision_recall_curve(y_training, y_scores)
# helpers.plot_precision_recall_curve(precisions, recalls)
# helpers.plot_precision_recall_vs_threshold(precisions, recalls, thresholds)

#results visualization - Receiver Operating Characteristic - training data
# fpr, tpr, thresholds = roc_curve(y_training, y_scores)
# helpers.plot_roc_curve(fpr, tpr)
