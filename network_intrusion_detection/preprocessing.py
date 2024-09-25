import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample

RANDOM_STATE = 43

label_encoder = LabelEncoder()


def get_simple_split(directory: str) -> (pd.DataFrame, pd.DataFrame):
    train = []
    test = []

    for filename in os.listdir(directory):
        if filename.startswith(('Monday', 'Tuesday', 'Wednesday')):
            train.append(pd.read_csv(os.path.join(directory, filename)))
        elif filename.startswith(('Thursday', 'Friday')):
            test.append(pd.read_csv(os.path.join(directory, filename)))

    return (pd.concat(train, ignore_index=True),
            pd.concat(test, ignore_index=True))


def get_mode_split(directory: str, splitmode: float) -> (pd.DataFrame, pd.DataFrame):
    df_list = []
    for filename in os.listdir(directory):
        df_list.append(pd.read_csv(os.path.join(directory, filename)))

    dframe = pd.concat(df_list, ignore_index=True)

    dframe = dframe.sample(frac=1, random_state=RANDOM_STATE)  # Shuffle dataset
    train_size = int(len(dframe) * splitmode)
    train = dframe[0:train_size]
    test = dframe[train_size:]

    return train, test


def change_label(label: str):
    if label == 'BENIGN':
        return 'Benign'
    elif 'DoS' in label or 'DDoS' in label:
        return 'DoS'
    elif label == 'PortScan':
        return 'Scan'
    else:
        return 'Exploit'


def undersample(dframe: pd.DataFrame, magnitude: float) -> pd.DataFrame:
    """
    Performs undersampling of "Benign" entries in the given DataFrame.

    :param dframe: The DataFrame to perform undersampling on.
    :param magnitude: The amount of rows to keep, should be between 0 and 1.
    :return: A new DataFrame containing the undersampled entries.
    """
    y = dframe["Label"]
    benign = dframe[y == "Benign"]
    others = dframe[y != "Benign"]

    benign_undersampled = resample(benign, replace=False,
                    n_samples=int(len(benign) * magnitude),
                    random_state=RANDOM_STATE)

    return pd.concat([benign_undersampled, others])


def preprocess(dframe: pd.DataFrame) -> pd.DataFrame:
    dframe.columns = dframe.columns.str.strip()
    dframe["Label"] = dframe["Label"].map(change_label) # Remap labels

    dframe = undersample(dframe, 0.25)
    dframe["Label"] = label_encoder.fit_transform(dframe["Label"])

    dframe.replace([np.inf, -np.inf], np.nan, inplace=True) # Replace infinite values

    return dframe


def get_dataset(directory: str, splitmode: float = 0.6) -> (pd.DataFrame, pd.DataFrame):
    """
    Retrieves the dataset from the given directory as pandas dataframes. The dataset is split
    into training and test sets according to the ratio specified by splitmode.

    :param directory: The directory with the dataset files.
    :param splitmode: Specified how to split the dataset. If 0 < splitmode < 1 then it is treated
            as a split ratio, otherwise the function puts data from Monday to Wednesday into the train set
            and data from Thursday to Friday into the test set.
    :return: A tuple of the form (train, test).
    """
    train, test = get_simple_split(directory)

    train = preprocess(train)
    test = preprocess(test)

    if 0 < splitmode < 1:
        combined = pd.concat([train, test], ignore_index=True)
        combined = combined.loc[:, (combined != 0).any(axis=0)] # Remove columns with all zeroes
        train, test = train_test_split(combined, train_size=splitmode,
                                       random_state=RANDOM_STATE, stratify=combined["Label"])

    return train, test


def main():
    train, test = get_dataset('MachineLearningCVE', 0.6)
    print(train.info())
    print(test.info())
    print(f'Train:\n{train["Label"].value_counts(normalize=True)}')
    print(f'Test:\n{test["Label"].value_counts(normalize=True)}')


if __name__ == '__main__':
    main()