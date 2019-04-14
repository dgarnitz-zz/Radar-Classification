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

X_mean = X_raw_data.loc[:,:255] # this takes only the means

#explore the data
helpers.checkDataForNullAndType(X_raw_data, y_raw_data)

#remove the training set
X_training, X_testing, y_training, y_testing = train_test_split(X_raw_data, y_raw_data, test_size = 0.2, random_state = 78)

#visualize the data
helpers.visualizeOneRowOfData(X_training)
helpers.visualizeOneRowOfData(X_mean)
helpers.visualizeStandardDeviation(X_training)
helpers.visualizeAllRowsOfData(X_raw_data)

#heatmap
# helpers.correlationMatrix(X_training)

#load into numpy array
X = X_training.values
y = y_training.values

#set classifcation
y_train = (y_training == 0)
y_test = (y_testing == 0)

#shuffle testing - use the data size, which is 64 when 20% of the data is left for testing
shuffle_index = np.random.permutation(64)
x_train, y_train = X_training[shuffle_index], y_train[shuffle_index]

#initialize the model - stochasic gradient descent classifier
sgd_clf = SGDClassifier(random_state=45, max_iter=1000, tol=1e-3)

#standardize the data
scaler = StandardScaler()
scaler.fit(x_train)  # Don't cheat - fit only on training data
x_train = scaler.transform(x_train)
X_testing = scaler.transform(X_testing)

#train the model
sgd_clf.fit(x_train, y_train.ravel())

#cross validation - use cross_val_predict to give the actual values
y_train_prediction = cross_val_predict(sgd_clf, x_train, y_train.ravel(), cv=5)

#caclulate the score for each training instance, then use it to plot Precision-Recall Curve and Receiver Operating Characteristic
y_scores = cross_val_predict(sgd_clf, x_train, y_train.ravel(), cv=5, method="decision_function")

#performance evaluation
print(confusion_matrix(y_train, y_train_prediction))
print("Precision is: ")                                 #True Positive / (True Positive + False Positive)
print(precision_score(y_train, y_train_prediction))
print("Recall is: ")                                    #True Positive / (True Positive + False Negative)
print(recall_score(y_train, y_train_prediction))
print("F1 Score is: ")                                  #useful for comparing two classifiers
print(f1_score(y_train, y_train_prediction))

#results visualization - Precision-Recall Curve
precisions, recalls, thresholds = precision_recall_curve(y_training, y_scores)
helpers.plot_precision_recall_curve(precisions, recalls)
helpers.plot_precision_recall_vs_threshold(precisions, recalls, thresholds)

#results visualization - Receiver Operating Characteristic
fpr, tpr, thresholds = roc_curve(y_training, y_scores)
helpers.plot_roc_curve(fpr, tpr)
