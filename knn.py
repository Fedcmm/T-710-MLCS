from sklearn.neighbors import KNeighborsClassifier

from preprocessing import create_test_data, create_train_data
from metrics import print_metrics
from vocabulary import get_most_frequent_words

k_values = [4, 6, 8, 10, 15, 20]

def main():
    for k in k_values:
        print(f'\n\n===== Test with k set to {k} =====')

        vocabulary = get_most_frequent_words('train-mails')["word"]

        x_train, y_train = create_train_data(vocabulary)
        x_test, y_test = create_test_data(vocabulary)

        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(x_train, y_train)

        print_metrics(knn, x_test, y_test)


if __name__ == '__main__':
    main()