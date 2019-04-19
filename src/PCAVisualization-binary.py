import numpy as np
import pandas as pd
import helpers
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict, GridSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, precision_recall_curve, roc_curve
from sklearn.pipeline import Pipeline

#load data
X_raw_data = pd.read_csv('../data/binary/X.csv', header=None)
y_raw_data = pd.read_csv('../data/binary/y.csv', header=None)

#remove the training set
X_training, X_testing, y_training, y_testing = train_test_split(X_raw_data, y_raw_data, test_size = 0.2, random_state = 78, stratify=y_raw_data)

#sort the data
X_training = X_training.sort_index()
y_training = y_training.sort_index()

#standardize the data
X = StandardScaler().fit_transform(X_training)

#Principal component analysis
pca = PCA(n_components=2)
data = pca.fit_transform(X)

principalDf = pd.DataFrame(data = data, columns = ['principal component 1', 'principal component 2'])

X1_y0 = principalDf.iloc[0:32,0]
X2_y0 = principalDf.iloc[0:32,1]
X1_y1 = principalDf.iloc[32:64,0]
X2_y1 = principalDf.iloc[32:64,1]

plt.scatter(X1_y0, X2_y0, color='green')
plt.scatter(X1_y1, X2_y1, color='blue')
plt.xlabel('X1')
plt.ylabel('X2')
plt.title("Principal Component Analysis - Two Features - Binary Data")
plt.show()

##############################################################################
#Verify that the PCA Model is acceptable
#############################################################################

#initialize the model
log_reg = LogisticRegression(solver='liblinear') #good choice for small datasets

#create pipeline & grid
pipeline = Pipeline([('model', log_reg)])

grid = [{'model__penalty': ['l1', 'l2'],
        'model__tol': [1e-3, 1e-4, 1e-5],
        'model__max_iter': [100, 500]}]

#Perform Grid Search and Cross Validation
clf = GridSearchCV(pipeline, param_grid = grid, cv=5, refit = True)

#train the model
clf.fit(data, y_training)

#cross validation - use cross_val_predict to give the actual values
y_train_prediction = cross_val_predict(clf, data, y_training, cv=5)

#caclulate the score for each training instance, then use it to plot Precision-Recall Curve and Receiver Operating Characteristic
y_scores = cross_val_predict(clf, data, y_training, cv=5, method="decision_function")

#performance evaluation of training data
print(confusion_matrix(y_training, y_train_prediction))
print("Precision is: ")                                 #True Positive / (True Positive + False Positive)
print(precision_score(y_training, y_train_prediction))
print("Recall is: ")                                    #True Positive / (True Positive + False Negative)
print(recall_score(y_training, y_train_prediction))
print("F1 Score is: ")                                  #useful for comparing two classifiers
print(f1_score(y_training, y_train_prediction))

#results visualization - Precision-Recall Curve - training data
precisions, recalls, thresholds = precision_recall_curve(y_training, y_scores)
helpers.plot_precision_recall_curve(precisions, recalls)
helpers.plot_precision_recall_vs_threshold(precisions, recalls, thresholds)

#results visualization - Receiver Operating Characteristic - training data
fpr, tpr, thresholds = roc_curve(y_training, y_scores)
helpers.plot_roc_curve(fpr, tpr)
