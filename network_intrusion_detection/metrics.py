import numpy as np
import seaborn as sb

from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(y_true, y_pred, labels, model_name: str, outfile: str):
    m = confusion_matrix(y_true, y_pred, labels=labels)
    ax = sb.heatmap(m, annot=True, cmap='Blues')
    ax.set_title(f'{model_name} Confusion matrix')
    ax.set_xlabel('Predicted Values')
    ax.set_ylabel('True Values')
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    plt.savefig(outfile)
    plt.show()

def plot_feature_importance(importance, feature_names, model_name: str, outfile: str):
    indices = np.argsort(importance)

    plt.title(f"{model_name} Feature Importance")
    plt.barh(range(len(indices)), importance[indices], color='b', align='center')
    plt.yticks(range(len(indices)), feature_names[indices])
    plt.xlabel("Relative Importance")
    plt.tight_layout()
    plt.savefig(outfile)
    plt.show()