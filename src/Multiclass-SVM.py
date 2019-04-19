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
X_raw_data = pd.read_csv('../data/multiclass/X.csv', header=None)
y_raw_data = pd.read_csv('../data/multiclass/y.csv', header=None)

X_means = X_raw_data.loc[:,:255] # this takes only the means

#explore the data
helpers.checkDataForNullAndType(X_raw_data, y_raw_data)

#remove the training set
X_training, X_testing, y_training, y_testing = train_test_split(X_raw_data, y_raw_data, test_size = 0.2, random_state = 78, stratify=y_raw_data)

#create a training and testing dataset from the mean values
X_training_means, X_testing_means, y_training_means, y_testing_means = train_test_split(X_means, y_raw_data, test_size = 0.2, random_state = 78, stratify=y_raw_data)

#create a training and testing dataset from the mean values
X_training_visualize, X_testing_visualize, y_training_visualize, y_testing_visualize = train_test_split(X_training, y_training, test_size = 0.75, random_state = 55, stratify=y_training)

# #visualize the data
# helpers.visualizeOneRowOfData(X_training)
# helpers.visualizeOneRowOfData(X_training_means)
# helpers.visualizeStandardDeviation(X_training)
# helpers.visualizeAllRowsOfData(X_training_visualize)
# helpers.histogram(y_training, 7, 'Histogram of Y Values For Binary Classification, With 6 Bins')
#
# #heatmap
# helpers.correlationMatrix(X_training_means, "Heatmap of Multiclass X Means Training Data")
# helpers.correlationMatrix(X_training, "Heatmap of Multiclass X Training Data")

#initialize the model
svm = SVC(kernel = 'rbf')

#standardize the data
scaler = preprocessing.StandardScaler().fit(X_training)

#create pipeline & grid
pipeline = Pipeline([('scaler', scaler),
        ('model', svm)])

grid = [{'model__tol': [1e-5],
        'model__max_iter': [1000]}]

#Perform Grid Search and Cross Validation
clf = GridSearchCV(pipeline, param_grid = grid, cv=5, refit = True)

#train the model
clf.fit(X_training, y_training)

#cross validation - use cross_val_predict to give the actual values
y_train_prediction = cross_val_predict(clf, X_testing, y_testing, cv=5)

#calculate the score for each training instance, then use it to plot Precision-Recall Curve and Receiver Operating Characteristic
y_scores = cross_val_predict(clf, X_testing, y_testing, cv=5, method="decision_function")

#confusion matrix and visualization
confusion_matrix = confusion_matrix(y_testing, y_train_prediction)
print(confusion_matrix)
xlabels=["air", "book", "hand", "knife", "plastic case"]
ylabels=["air", "book", "hand", "knife", "plastic case"]
helpers.confusionMatrix(confusion_matrix, xlabels, ylabels, "Multiclass SVM Confusion Matrix")

#performance evaluation of training data - per class
print("Precision is: ")                                 #True Positive / (True Positive + False Positive)
print(precision_score(y_testing, y_train_prediction, average='macro'))
print("Recall is: ")                                    #True Positive / (True Positive + False Negative)
print(recall_score(y_testing, y_train_prediction, average='macro'))
print("F1 Score is: ")                                  #useful for comparing two classifiers
print(f1_score(y_testing, y_train_prediction, average='macro'))

#write the unclassified data to a file
X_to_classify = pd.read_csv('../data/multiclass/XToClassify.csv', header=None)
scaler = StandardScaler()
scaler.fit(X_to_classify)
X_to_classify = scaler.transform(X_to_classify)
y_prediction = clf.predict(X_to_classify)
y_prediction_string = np.array2string(y_prediction)
print(y_prediction_string)
text_file = open("PredictedClasses.csv", "a")
text_file.write(y_prediction_string)
text_file.close()
