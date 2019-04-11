import matplotlib.pyplot as plt

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
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g--", label="Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="center left")
    plt.ylim([0, 1])
    plt.title("Precision-Recall-Thresholds Curve")
    plt.show()

#from "Hands-on Machine Learning with Scikit-Learn & TensorFlow" by Aurelien Geron, page 93
def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title("Receiver Operating Characteristic Curve")
    plt.show()

#precision recall curve
def plot_precision_recall_curve(precisions, recalls): 
    plt.plot(recalls[:-1], precisions[:-1], "b--")
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0, 1])
    plt.title("Precision-Recall Curve")
    plt.show()

#Visualize one row of data
def visualizeOneRowOfData(X):
    plt.plot(X.columns, X.loc[0,:])
    plt.xlabel('Feature Number')
    plt.ylabel('Feature value')
    plt.title("One Row of Data Visualized")
    plt.show()

#visualize standard deviation
def visualizeStandardDeviation(X):
    array = []
    for i in X.columns:
        array.append(X.loc[:, i].std()) 
    plt.plot(X.columns, array)
    plt.xlabel('Column Number')
    plt.ylabel('Standard Deivation')
    plt.title("Standard Deviation of Each Row")
    plt.show()

#compare standard deviation of the classes --> this does not work
# def compareSD(X, y): 
#     yEquals0 = []
#     yEquals1 = []
#     for i in X.columns:
#         if y.loc[i,0] == 0:
            
#         array.append(X.loc[:, i].std()) 
#     print("Porco Dio : " )
#     plt.plot(X.columns, array)
#     plt.show()