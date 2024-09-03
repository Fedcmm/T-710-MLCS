import time

from sklearn.naive_bayes import MultinomialNB

from utils.metrics import print_metrics, plot_roc_curves
from utils.preprocessing import create_test_data, create_train_data
from utils.vocabulary import get_most_frequent_words

vocabulary_sizes = [100, 500, 1000, 2000, 3000]

def compare_sizes():
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
        ys_test, ys_pred,
        'bayes.png'
    )

def train_bayes(x_train, y_train) -> MultinomialNB:
    bayes = MultinomialNB()
    # noinspection PyTypeChecker
    return bayes.fit(x_train, y_train)

if __name__ == '__main__':
    compare_sizes()