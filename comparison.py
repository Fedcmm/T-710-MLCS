from bayes import train_bayes
from knn import train_knn
from logistic import train_logistic
from utils.metrics import print_metrics, plot_roc_curves
from utils.preprocessing import create_test_data, create_train_data
from utils.vocabulary import get_most_frequent_words


def main():
    vocabulary = get_most_frequent_words('train-mails')["word"]
    x_train, y_train = create_train_data(vocabulary)
    x_test, y_test = create_test_data(vocabulary)

    models = {'Naive Bayes': train_bayes(x_train, y_train),
              'KNN': train_knn(x_train, y_train),
              'Logistic Regression': train_logistic(x_train, y_train)}

    ys_test = []
    ys_pred = []

    print('===== Hyper-parameter values =====')
    print('Vocabulary size: 3000')
    print('KNN k value: 10')
    print('Logistic Regression C value: 0.001', end='\n\n')

    for name, model in models.items():
        print(f'\n===== Results for {name} =====')
        print_metrics(model, x_test, y_test)
        ys_test.append(y_test)
        ys_pred.append(model.predict_proba(x_test)[:, 1])

    plot_roc_curves(
        'ROC curves for the different models',
        ['Bayes', 'KNN', 'Logistic'],
        ys_test, ys_pred,
        'comparison.png'
    )

if __name__ == '__main__':
    main()