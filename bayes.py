import time

from matplotlib import pyplot as plt
from seaborn import heatmap
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB

from utils.metrics import print_metrics, plot_roc_curves
from utils.preprocessing import create_test_data, create_train_data
from vocabulary import get_most_frequent_words

vocabulary_sizes = [100, 500, 1000, 2000, 3000]

def plot_confusion_matrix(y_test, y_pred):
    m = confusion_matrix(y_test, y_pred)
    ax = heatmap(m, annot=True, cmap='Blues')
    ax.set_title('Confusion matrix\n')
    ax.set_xlabel('Predicted Values')
    ax.set_ylabel('True Values')
    ax.xaxis.set_ticklabels(['False', 'True'])
    ax.yaxis.set_ticklabels(['False', 'True'])
    plt.show()

def test_sizes():
    ys_test = []
    ys_pred = []
    for size in vocabulary_sizes:
        print(f'\n===== Test with vocabulary size {size} =====')

        vocabulary = get_most_frequent_words('train-mails', size)["word"]

        x_train, y_train = create_train_data(vocabulary)
        x_test, y_test = create_test_data(vocabulary)

        bayes = MultinomialNB()
        ts = time.time()
        bayes.fit(x_train, y_train)
        print(f'Training time: {time.time() - ts}')

        print_metrics(bayes, x_test, y_test)

        ys_test.append(y_test)
        ys_pred.append(bayes.predict_proba(x_test)[:, 1])

    plot_roc_curves(
        'ROC Curves for different vocabulary sizes',
        vocabulary_sizes,
        ys_test, ys_pred
    )

def train_bayes(x_train, y_train) -> MultinomialNB:
    bayes = MultinomialNB()
    # noinspection PyTypeChecker
    return bayes.fit(x_train, y_train)

def main():
    test_sizes()

if __name__ == '__main__':
    main()