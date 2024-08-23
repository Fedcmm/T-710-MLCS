import os
from pandas import DataFrame


def extract_words(file_contents: str) -> list:
    return filter(lambda s: s.isalpha(), file_contents.split(" "))

def count_words(words: list) -> dict[str, int]:
    count_dict = {}
    for word in words:
        count_dict[word] = count_dict.get(word, 0) + 1
    return count_dict

def merge_dicts(dict1: dict, dict2: dict) -> dict:
    for key, value in dict2.items():
        dict1[key] = dict1.get(key, 0) + value
    return dict1

def get_most_frequent_words(directory: str) -> DataFrame:
    """
    Return the 2000 most frequent worlds that appear in the files contained in the
    given directory

    :param directory The directory to search files in
    """
    total_count = {}
    for f_name in os.listdir(directory):
        with open(f"{directory}/{f_name}") as f:
            words = extract_words(f.read().splitlines()[2])
            count_dict = count_words(words)
            total_count = merge_dicts(total_count, count_dict)

    #return list(dict(sorted(total_count.items(), key=lambda x: x[1], reverse=True)).keys())[:2000]

    count_df = DataFrame(list(total_count.items()), columns=["word", "count"])
    #count_df = pd.DataFrame.from_dict(total_count, orient='index', columns=['count'])
    count_df.sort_values(by=['count'], ascending=False, inplace=True)
    count_df.reset_index(drop=True, inplace=True)
    return count_df.head(2000)

def main():
    print(get_most_frequent_words("train-mails"))

if __name__ == '__main__':
    main()