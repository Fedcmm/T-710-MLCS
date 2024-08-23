import os
import re

import pandas as pd
from matplotlib import pyplot as plt
from seaborn import heatmap
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import precision_score, confusion_matrix
from sklearn.naive_bayes import MultinomialNB

from dictionary import get_most_frequent_words


def get_labels(directory) -> list:
    labels = []
    for f_name in os.listdir(directory):
        labels.append(1 if re.match('spmsg.{2,}.txt', f_name) else 0)
    return labels

def read_files(directory) -> list:
    files = []
    for f_name in os.listdir(directory):
        with open(f"{directory}/{f_name}", 'r') as f:
            files.append(f.readlines()[2])

    return files

def create_data(directory) -> pd.DataFrame:
    vocabulary = get_most_frequent_words('train-mails')["word"]
    vectorizer = CountVectorizer(input='filename', vocabulary=vocabulary)

    f_names = [f"{directory}/{f_name}" for f_name in os.listdir(directory)]
    count_vector = vectorizer.fit_transform(f_names)

    return pd.DataFrame(
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

def main():
    x_train = create_data('train-mails')
    y_train = get_labels('train-mails')
    x_test = create_data('test-mails')
    y_test = get_labels('test-mails')

    print(x_train.shape)
    print(x_test.shape)

    model = MultinomialNB()
    model.fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)
    print(f"Accuracy: {accuracy}")

    y_pred = model.predict(x_test)
    print(precision_score(y_test, y_pred))

    show_confusion_matrix(y_test, y_pred)

if __name__ == '__main__':
    main()