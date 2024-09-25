import numpy as np
import seaborn as sb

from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, recall_score, f1_score


def plot_confusion_matrix(y_true, y_pred):
    m = confusion_matrix(y_true, y_pred)
    ax = sb.heatmap(m, annot=True, cmap='Blues')
    ax.set_title('Confusion matrix')
    ax.set_xlabel('Predicted Values')
    ax.set_ylabel('True Values')
    plt.show()


def display_metrics(y_true, y_pred):
    print(classification_report(y_true, y_pred))
    #print(f'Accuracy: {accuracy_score(y_true, y_pred)}')
    #print(f'Recall: {recall_score(y_true, y_pred, average='macro')}')
    #print(f'F1-Score: {f1_score(y_true, y_pred)}')
    plot_confusion_matrix(y_true, y_pred)


def plot_feature_importance(importance, feature_names):
    indices = np.argsort(importance)

    plt.title("Feature Importance")
    plt.barh(range(len(indices)), importance[indices], color='b', align='center')
    plt.yticks(range(len(indices)), feature_names[indices])
    plt.xlabel("Relative Importance")
    plt.show()