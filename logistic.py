from sklearn.linear_model import LogisticRegression

from utils.metrics import print_metrics, plot_roc_curves
from utils.preprocessing import create_train_data, create_test_data
from vocabulary import get_most_frequent_words

c_values = [0.001, 0.4, 1, 4, 9]

def compare_c_values():
    vocabulary = get_most_frequent_words('train-mails')["word"]
    ys_test = []
    ys_pred = []

    x_train, y_train = create_train_data(vocabulary)
    x_test, y_test = create_test_data(vocabulary)

    for c in c_values:
        print(f'\n===== Test with regularization set to {c} =====')

        logistic = train_logistic(x_train, y_train, c)

        print('Results for test set:', end='\t')
        print_metrics(logistic, x_test, y_test)
        print('Results for train set:', end='\t')
        print_metrics(logistic, x_train, y_train)

        ys_test.append(y_test)
        ys_pred.append(logistic.predict_proba(x_test)[:, 1])

    plot_roc_curves(
        'ROC Curves for different C values',
        c_values,
        ys_test, ys_pred
    )

def train_logistic(x_train, y_train, c: float = 4) -> LogisticRegression:
    logistic = LogisticRegression(C=c)
    return logistic.fit(x_train, y_train)

if __name__ == '__main__':
    compare_c_values()