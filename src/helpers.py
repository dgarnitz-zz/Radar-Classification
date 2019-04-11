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
def plot_precision_recall_vs__threshold(precisions, recalls, thresholds): 
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g--", label="Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="center left")
    plt.ylim([0, 1])
    plt.show()

#from "Hands-on Machine Learning with Scikit-Learn & TensorFlow" by Aurelien Geron, page 93
def plot_roc_curve(fpr, tpr, label=None):
        plt.plot(fpr, tpr, linewidth=2, label=label)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.axis([0, 1, 0, 1])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.show()

def plot_precision_recall_curve(precisions, recalls): 
    plt.plot(recalls[:-1], precisions[:-1], "b--")
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc="center left")
    plt.ylim([0, 1])
    plt.show()