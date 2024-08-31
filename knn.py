from sklearn.neighbors import KNeighborsClassifier

from preprocessing import create_test_data, create_train_data
from metrics import print_metrics, plot_roc_curves
from vocabulary import get_most_frequent_words

k_values = [4, 6, 8, 10, 15, 20]

def test_k_values():
    vocabulary = get_most_frequent_words('train-mails')["word"]
    ys_test = []
    ys_pred = []

    for k in k_values:
        print(f'\n===== Test with k set to {k} =====')

        x_train, y_train = create_train_data(vocabulary)
        x_test, y_test = create_test_data(vocabulary)

        knn = train_knn(x_train, y_train, k)

        print_metrics(knn, x_test, y_test)

        ys_test.append(y_test)
        ys_pred.append(knn.predict(x_test))

    plot_roc_curves(
        'ROC Curves for different k values',
        k_values,
        ys_test, ys_pred
    )

def train_knn(x_train, y_train, k = 15) -> KNeighborsClassifier:
    knn = KNeighborsClassifier(n_neighbors=k)
    return knn.fit(x_train, y_train)

if __name__ == '__main__':
    test_k_values()