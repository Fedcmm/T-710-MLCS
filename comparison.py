from bayes import train_bayes
from knn import train_knn
from logistic import train_logistic
from metrics import print_metrics, plot_roc_curves
from preprocessing import create_test_data, create_train_data
from vocabulary import get_most_frequent_words


def main():
    vocabulary = get_most_frequent_words('train-mails')["word"]
    x_train, y_train = create_train_data(vocabulary)
    x_test, y_test = create_test_data(vocabulary)

    models = [train_bayes(x_train, y_train),
              train_knn(x_train, y_train),
              train_logistic(x_train, y_train)]

    ys_test = []
    ys_pred = []

    for model in models:
        print_metrics(model, x_test, y_test)
        ys_test.append(y_test)
        ys_pred.append(model.predict(x_test))

    plot_roc_curves(
        'ROC curves for the different models',
        ['Bayes', 'KNN', 'Logistic'],
        ys_test, ys_pred
    )

if __name__ == '__main__':
    main()