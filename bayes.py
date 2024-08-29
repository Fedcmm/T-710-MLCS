import os
import time

from pandas import DataFrame
from matplotlib import pyplot as plt
from seaborn import heatmap
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import precision_score, confusion_matrix
from sklearn.naive_bayes import MultinomialNB

from vocabulary import get_most_frequent_words


vocabulary_sizes = [100, 500, 1000, 2000, 3000]

def get_labels(directory) -> list:
    labels = []
    for f_name in os.listdir(directory):
        labels.append(1 if f_name.startswith("spmsg") else 0)
    return labels

def create_data(directory, vocabulary) -> DataFrame:
    vectorizer = CountVectorizer(input='filename', vocabulary=vocabulary)

    f_names = [f"{directory}/{f_name}" for f_name in os.listdir(directory)]
    count_vector = vectorizer.fit_transform(f_names)

    return DataFrame(
        data=count_vector.toarray(),
        columns=vectorizer.vocabulary,
        index=f_names
    )

def show_confusion_matrix(y_test, y_pred):
    m = confusion_matrix(y_test, y_pred)
    ax = heatmap(m, annot=True, cmap='Blues')
    ax.set_title('Confusion matrix\n')
    ax.set_xlabel('Predicted Values')
    ax.set_ylabel('True Values')
    ax.xaxis.set_ticklabels(['False', 'True'])
    ax.yaxis.set_ticklabels(['False', 'True'])
    plt.show()

def test_sizes():
    for size in vocabulary_sizes:
        print(f'\n\n===== Test with vocabulary size {size} =====')

        ts = time.time()
        vocabulary = get_most_frequent_words('train-mails', size)["word"]

        x_train = create_data('train-mails', vocabulary)
        y_train = get_labels('train-mails')
        x_test = create_data('test-mails', vocabulary)
        y_test = get_labels('test-mails')

        bayes = MultinomialNB()
        bayes.fit(x_train, y_train)
        print(f'Elapsed time: {time.time() - ts}')

        accuracy = bayes.score(x_test, y_test)
        print(f"Accuracy: {accuracy}")

        y_pred = bayes.predict(x_test)
        print(f'Precision: {precision_score(y_test, y_pred)}')

def main():
    test_sizes()

if __name__ == '__main__':
    main()