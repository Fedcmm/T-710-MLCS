import numpy as np
import seaborn as sb

from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(y_true, y_pred, outfile):
    m = confusion_matrix(y_true, y_pred)
    ax = sb.heatmap(m, annot=True, cmap='Blues')
    ax.set_title('Confusion matrix')
    ax.set_xlabel('Predicted Values')
    ax.set_ylabel('True Values')
    plt.savefig(outfile)
    plt.show()


#def display_metrics(y_true, y_pred):
#    print(classification_report(y_true, y_pred, digits=4))
#    plot_confusion_matrix(y_true, y_pred)


def plot_feature_importance(importance, feature_names, outfile):
    indices = np.argsort(importance)

    plt.title("Feature Importance")
    plt.barh(range(len(indices)), importance[indices], color='b', align='center')
    plt.yticks(range(len(indices)), feature_names[indices])
    plt.xlabel("Relative Importance")
    plt.tight_layout()
    plt.savefig(outfile)
    plt.show()