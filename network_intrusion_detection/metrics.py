import numpy as np
import seaborn as sb

from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.tree import plot_tree


def plot_confusion_matrix(y_true, y_pred, outfile: str):
    m = confusion_matrix(y_true, y_pred)
    ax = sb.heatmap(m, annot=True, cmap='Blues')
    ax.set_title('Confusion matrix')
    ax.set_xlabel('Predicted Values')
    ax.set_ylabel('True Values')
    plt.savefig(outfile)
    plt.show()


def plot_decision_tree(tree, feature_names, class_names, outfile):
    plt.figure(figsize=(25, 20))
    plot_tree(tree, feature_names=feature_names, class_names=class_names, fontsize=13, max_depth=4)
    plt.savefig(outfile)
    plt.show()

def plot_feature_importance(importance, feature_names, outfile: str):
    indices = np.argsort(importance)

    plt.title("Feature Importance")
    plt.barh(range(len(indices)), importance[indices], color='b', align='center')
    plt.yticks(range(len(indices)), feature_names[indices])
    plt.xlabel("Relative Importance")
    plt.tight_layout()
    plt.savefig(outfile)
    plt.show()