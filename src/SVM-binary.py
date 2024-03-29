import numpy as np
import pandas as pd
import helpers
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict, GridSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, precision_recall_curve, roc_curve
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


#load data
X_raw_data = pd.read_csv('../data/binary/X.csv', header=None)
y_raw_data = pd.read_csv('../data/binary/y.csv', header=None)

#remove the training set
X_training, X_testing, y_training, y_testing = train_test_split(X_raw_data, y_raw_data, test_size = 0.2, random_state = 78, stratify=y_raw_data)

#initialize the model - Support vector classification
svc = SVC()

#standardize the data
scaler = preprocessing.StandardScaler().fit(X_training)

#create pipeline & grid
pipeline = Pipeline([('scaler', scaler),
        ('model', svc)])

grid = [{'model__kernel': ['linear'],
        'model__tol': [1e-5]}]

#Perform Grid Search and Cross Validation
clf = GridSearchCV(pipeline, param_grid = grid, cv=5, refit = True)

#train the model
clf.fit(X_training, y_training)

#cross validation - use cross_val_predict to give the actual values
y_train_prediction = cross_val_predict(clf, X_testing, y_testing, cv=5)

#calculate the score for each training instance, then use it to plot Precision-Recall Curve and Receiver Operating Characteristic
y_scores = cross_val_predict(clf, X_testing, y_testing, cv=5, method="decision_function")

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
title = "SVM Confusion Matrix"
helpers.confusionMatrix(confusion_matrix, xlabels, ylabels, title)

#results visualization - Precision-Recall Curve
precisions, recalls, thresholds = precision_recall_curve(y_testing, y_scores)
helpers.plot_precision_recall_curve(precisions, recalls)
helpers.plot_precision_recall_vs_threshold(precisions, recalls, thresholds)

#results visualization - Receiver Operating Characteristic
fpr, tpr, thresholds = roc_curve(y_testing, y_scores)
helpers.plot_roc_curve(fpr, tpr)

#write the unclassified data to a file
X_to_classify = pd.read_csv('../data/binary/XToClassify.csv', header=None)
scaler = StandardScaler()
scaler.fit(X_to_classify)
X_to_classify = scaler.transform(X_to_classify)
y_prediction = clf.predict(X_to_classify)
y_prediction_string = np.array2string(y_prediction)
text_file = open("PredictedClasses.csv", "a")
text_file.write(y_prediction_string)
text_file.close()
