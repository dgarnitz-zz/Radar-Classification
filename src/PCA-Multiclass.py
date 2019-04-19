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

#load data
X_raw_data = pd.read_csv('../data/multiclass/X.csv', header=None)
y_raw_data = pd.read_csv('../data/multiclass/y.csv', header=None)

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
X1_y2 = principalDf.iloc[64:96,0]
X2_y2 = principalDf.iloc[64:96,1]
X1_y3 = principalDf.iloc[96:128,0]
X2_y3 = principalDf.iloc[96:128,1]
X1_y4 = principalDf.iloc[128:160,0]
X2_y4 = principalDf.iloc[128:160,1]

plt.scatter(X1_y0, X2_y0, color='orange') #air
plt.scatter(X1_y1, X2_y1, color='blue')   #book
plt.scatter(X1_y2, X2_y2, color='green')  #hand
plt.scatter(X1_y3, X2_y3, color='purple') #knife
plt.scatter(X1_y4, X2_y4, color='red')    #plastic case
plt.xlabel('X1')
plt.ylabel('X2')
plt.title("Principal Component Analysis - Two Features - Multiclass Data")
plt.show()
