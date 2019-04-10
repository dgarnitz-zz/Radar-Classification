import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from SGDBinary import SGDBinary

print("Perfomring Binary Classification")
print("Loading and cleaning data")

#load data
X_raw_data = pd.read_csv('../data/binary/X.csv', header=None)
y_raw_data = pd.read_csv('../data/binary/y.csv', header=None)

#check for null
if(not X_raw_data.isnull().any().any() and not y_raw_data.isnull().any().any()):
    print("There are no nul values in this dataset")

#load into numpy array
X = X_raw_data.values
y = y_raw_data.values 

#remove the training set 
X_training, X_testing, y_training, y_testing = train_test_split(X, y, test_size = 0.2, random_state = 78)

#prompt user to choose model
print("Please enter a number corresponding to the following options")
print("1 - SGD Classifier Model")
print("2 - Logistic Regression Model")

#read in user input
choice = input()

if(choice == "1"):
    SGD = SGDBinary(X_training, X_testing, y_training, y_testing)
    SGD.performanceEvaluation()
    SGD.visualizeResults()        