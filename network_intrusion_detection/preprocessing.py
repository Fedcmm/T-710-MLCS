import os

import pandas as pd
from sklearn.utils import resample

RANDOM_STATE = 43


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


def get_dataset(directory: str, splitmode: float = 0.6) -> (pd.DataFrame, pd.DataFrame):
    df_list = []
    for filename in os.listdir(directory):
        df_list.append(pd.read_csv(os.path.join(directory, filename)))

    dframe = pd.concat(df_list, ignore_index=True)
    dframe.columns = dframe.columns.str.strip()
    dframe["Label"] = dframe["Label"].map(change_label) # Remap labels

    dframe = dframe.loc[:, (dframe != 0).any(axis=0)] # Remove columns with all zeroes
    dframe = undersample(dframe, 0.25)

    dframe = dframe.sample(frac=1, random_state=RANDOM_STATE) # Shuffle dataset
    train_size = int(len(dframe) * splitmode)
    train = dframe[0:train_size]
    test = dframe[train_size:]

    return train, test


def main():
    train, test = get_dataset('MachineLearningCVE')
    print(f'Train:\n{train["Label"].value_counts(normalize=True)}')
    print(f'Test:\n{test["Label"].value_counts(normalize=True)}')

if __name__ == '__main__':
    main()