from matplotlib import pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc


def print_metrics(model, x, y):
    """
    Prints different metrics for the given model. These are:
        * Accuracy
        * Precision
        * Recall
        * F1 Score

    :param model: The model to evaluate
    :param x: The test data
    :param y: The test labels
    """
    accuracy = model.score(x, y)
    y_pred = model.predict(x)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)

    print(f'Accuracy: {accuracy:.4f}',
          f'Precision: {precision_score(y, y_pred):.4f}',
          f'Recall: {recall:.4f}',
          f'F1 score: {f1:.4f}',
          sep='\t')

def plot_roc_curves(title: str, plt_labels: list, ys_test: list, ys_pred: list):
    plt.figure()

    for y_test, y_pred, label in zip(ys_test, ys_pred, plt_labels):
        fpr, tpr, _ = roc_curve(y_test, y_pred)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{label} - AUC = {roc_auc:.3f}')

    plt.plot([0, 1], [0, 1], 'k--',)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend()
    plt.show()