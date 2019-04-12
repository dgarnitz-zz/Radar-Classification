import numpy as np
import pandas as pd
import helpers
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#load data
X_raw_data = pd.read_csv('../data/binary/X.csv', header=None)
y_raw_data = pd.read_csv('../data/binary/y.csv', header=None)

#remove the training set 
X_training, X_testing, y_training, y_testing = train_test_split(X_raw_data, y_raw_data, test_size = 0.2, random_state = 78, stratify=y_raw_data)

#sort the data
X_training.sort_index()
y_training = y_training.sort_index()

#standardize the data
X = StandardScaler().fit_transform(X_training) 

#Principal component analysis
pca = PCA(n_components=2)
data = pca.fit_transform(X)

principalDf = pd.DataFrame(data = data, columns = ['principal component 1', 'principal component 2'])

X1 = principalDf.iloc[:,0]
X2 = principalDf.iloc[:,1]

plt.scatter(X1, X2)
plt.xlabel('X1')
plt.ylabel('X2')
plt.title("Principal Component Analysis - Two Features")
plt.show()






