import os

from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer


def create_labels(directory) -> list:
    labels = []
    for f_name in os.listdir(directory):
        labels.append(1 if f_name.startswith("spmsg") else 0)
    return labels

def create_data(directory, vocabulary) -> DataFrame:
    """
    Creates data suitable to be used with a model by reading files from the given directory.
    The words in the files are vectorized using a CountVectorizer initialised with the given vocabulary.

    :param directory: The directory containing the files to be processed
    :param vocabulary: The vocabulary to be used in the vectorization
    :return: A DataFrame containing the word counts, indexed with the file names
    """
    vectorizer = CountVectorizer(input='filename', vocabulary=vocabulary)

    f_names = [f"{directory}/{f_name}" for f_name in os.listdir(directory)]
    count_vector = vectorizer.fit_transform(f_names)

    return DataFrame(
        data=count_vector.toarray(),
        columns=vectorizer.vocabulary,
        index=f_names
    )

def create_train_data(vocabulary) -> (DataFrame, list):
    """
    Creates the training data by reading from the 'train-mails' directory of the dataset.

    :return: A DataFrame with the data and a list with its labels
    """
    return (create_data('train-mails', vocabulary),
            create_labels('train-mails'))

def create_test_data(vocabulary) -> (DataFrame, list):
    """
    Creates the test data by reading from the 'test-mails' directory of the dataset.

    :return: A DataFrame with the data and a list with its labels
    """
    return (create_data('test-mails', vocabulary),
            create_labels('test-mails'))