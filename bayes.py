import time

from matplotlib import pyplot as plt
from seaborn import heatmap
from sklearn.metrics import precision_score, confusion_matrix, roc_curve
from sklearn.naive_bayes import MultinomialNB

from preprocessing import create_test_data, create_train_data
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

def plot_roc_curve(ys_test: list, ys_pred: list):
    plt.figure()

    for y_test, y_pred, vocab_size in zip(ys_test, ys_pred, vocabulary_sizes):
        fpr, tpr, thresholds = roc_curve(y_test, y_pred)
        #roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'ROC curve size={vocab_size}')
    plt.plot([0, 1], [0, 1], 'k--',)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for different vocabulary sizes')
    plt.legend()
    plt.show()

def test_sizes():
    ys_test = []
    ys_pred = []
    for size in vocabulary_sizes:
        print(f'\n\n===== Test with vocabulary size {size} =====')

        ts = time.time()
        vocabulary = get_most_frequent_words('train-mails', size)["word"]

        x_train, y_train = create_train_data(vocabulary)
        x_test, y_test = create_test_data(vocabulary)

        bayes = MultinomialNB()
        bayes.fit(x_train, y_train)
        print(f'Elapsed time: {time.time() - ts}')

        accuracy = bayes.score(x_test, y_test)
        print(f"Accuracy: {accuracy}")

        y_pred = bayes.predict(x_test)
        print(f'Precision: {precision_score(y_test, y_pred)}')

        ys_test.append(y_test)
        ys_pred.append(y_pred)

    plot_roc_curve(ys_test, ys_pred)

def main():
    test_sizes()

if __name__ == '__main__':
    main()