import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

#method for exploring the data
def checkDataForNullAndType(X_raw_data, y_raw_data):

    #check for null
    if(not X_raw_data.isnull().any().any() and not y_raw_data.isnull().any().any()):
        print("There are no null values in this dataset")
    else:
        print("There are null values in this dataset")

    #check for string - every column appears to be of type 'float64', there are no strings
    print(X_raw_data.dtypes)
    print(X_raw_data.select_dtypes(include=[object]))

#from "Hands-on Machine Learning with Scikit-Learn & TensorFlow" by Aurelien Geron, page 91
def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=3.0)
    plt.plot(thresholds, recalls[:-1], "g--", label="Recall", linewidth=3.0)
    plt.xlabel("Threshold")
    plt.legend(loc="center left")
    plt.ylim([0, 1])
    plt.title("Precision-Recall-Thresholds Curve")
    plt.show()

#from "Hands-on Machine Learning with Scikit-Learn & TensorFlow" by Aurelien Geron, page 93
def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=5, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title("Receiver Operating Characteristic Curve")
    plt.show()

#precision recall curve
def plot_precision_recall_curve(precisions, recalls):
    plt.plot(recalls[:-1], precisions[:-1], "b--", linewidth=3.0)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0, 1])
    plt.title("Precision-Recall Curve")
    plt.show()

#visualize one row of data
def visualizeOneRowOfData(X):
    plt.plot(X.columns, X.loc[0,:])
    plt.xlabel('Feature Number')
    plt.ylabel('Feature Value')
    plt.title("One Row of Data Visualized")
    plt.show()

#visualize all rows of data
def visualizeAllRowsOfData(X):
    for index, row in X.iterrows():
        plt.plot(X.columns, X.loc[index,:])

    plt.title("All Data Points Visualized")
    plt.xlabel('Feature Number')
    plt.ylabel('Feature Value')
    plt.show()

#visualize standard deviation
def visualizeStandardDeviation(X):
    array = []
    for i in X.columns:
        array.append(X.loc[:, i].std())
    plt.plot(X.columns, array)
    plt.xlabel('Column Number')
    plt.ylabel('Standard Deivation')
    plt.title("Standard Deviation of Each Column")
    plt.show()

#heatmap for correlation matrix
def correlationMatrix(dataframe):
    corr = dataframe.corr()
    sns.heatmap(corr)
    plt.show()

#heatmap for confusion matrix
def confusionMatrix(data, xlabels, ylabels):
    sns.heatmap(data, annot=True, xticklabels=xlabels, yticklabels=ylabels, fmt='.0f')
    plt.show()

#histogram for y values
def histogram(y, bins, title):
    print(y.values)
    n, bins, patches = plt.hist(y.values, bins=bins, density = 0, facecolor = 'blue', alpha = .2)
    plt.title(title)
    plt.xlabel('Grouping of Y Values')
    plt.ylabel('Frequency')
    plt.show()
