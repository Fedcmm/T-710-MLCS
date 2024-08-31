import os
from pandas import DataFrame


def extract_words(file_contents: str) -> list[str]:
    # "Subject:" is filtered out by isalpha()
    return filter(lambda word: word.isalpha() and len(word) > 1, file_contents.split(" "))

def count_words(words: list) -> dict[str, int]:
    count_dict = {}
    for word in words:
        count_dict[word] = count_dict.get(word, 0) + 1
    return count_dict

def merge_dicts(dict1: dict, dict2: dict) -> dict:
    for key, value in dict2.items():
        dict1[key] = dict1.get(key, 0) + value
    return dict1

def count_all_words(directory: str) -> dict[str, int]:
    total_count = {}
    for f_name in os.listdir(directory):
        with open(f"{directory}/{f_name}") as f:
            words = extract_words(f.read())
            count_dict = count_words(words)
            total_count = merge_dicts(total_count, count_dict)
    return total_count

def get_most_frequent_words(directory: str, amount: int = 1000) -> DataFrame:
    """
    Return the most frequent words that appear in the files contained in the
    given directory

    :param directory: The directory to search files in
    :param amount: The amount of words to return
    :return: A DataFrame with the columns "word" and "count"
    """
    total_count = count_all_words(directory)

    count_df = DataFrame(list(total_count.items()), columns=["word", "count"])
    count_df.sort_values(by=['count'], ascending=False, inplace=True)
    count_df.reset_index(drop=True, inplace=True)
    return count_df.head(amount)

def main():
    print(get_most_frequent_words("train-mails"))

if __name__ == '__main__':
    main()