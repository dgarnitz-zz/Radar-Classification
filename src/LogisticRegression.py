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
X_raw_data = pd.read_csv('../data/binary/X.csv', header=None)
y_raw_data = pd.read_csv('../data/binary/y.csv', header=None)

#check for null
if(not X_raw_data.isnull().any().any() and not y_raw_data.isnull().any().any()):
    print("There are no nul values in this dataset")

#check for string - every column appears to be of type 'float64', there are no strings
#print(X_raw_data.dtypes)
#print(X_raw_data.select_dtypes(include=[object]))

#load into numpy array
X = X_raw_data.values
y = y_raw_data.values 

#remove the training set 
X_training, X_testing, y_training, y_testing = train_test_split(X, y, test_size = 0.2, random_state = 78)

#set classifcation
y_train = (y_training == 0)
y_test = (y_testing == 0)

#shuffle testing - use the data size, which is 64 when 20% of the data is left for testing
shuffle_index = np.random.permutation(64)
x_train, y_train = X_training[shuffle_index], y_train[shuffle_index]

#initialize the model - stochasic gradient descent classifier
log_reg = LogisticRegression(solver='liblinear') #good choice for small datasets

#standardize the data
scaler = preprocessing.StandardScaler().fit(x_train) 

 #create pipeline & grid
pipeline = Pipeline([('scaler', scaler), 
        ('model', log_reg)])

grid = [{'model__penalty': ['l1', 'l2'],
        'model__tol': [1e-3, 1e-4, 1e-5],
        'model__max_iter': [100, 500, 1000]}] 

#Perform Grid Search and Cross Validation
clf = GridSearchCV(pipeline, param_grid = grid, cv=5, refit = True)

#train the model
clf.fit(x_train, y_train.ravel())
#clf.refit  

#cross validation - use cross_val_predict to give the actual values
y_train_prediction = cross_val_predict(clf, x_train, y_train.ravel(), cv=5)

#caclulate the score for each training instance, then use it to plot Precision-Recall Curve and Receiver Operating Characteristic
y_scores = cross_val_predict(clf, x_train, y_train.ravel(), cv=5, method="decision_function")

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
helpers.plot_precision_recall_vs__threshold(precisions, recalls, thresholds)

#results visualization - Receiver Operating Characteristic
fpr, tpr, thresholds = roc_curve(y_training, y_scores)
helpers.plot_roc_curve(fpr, tpr)