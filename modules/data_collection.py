import json
import re
import os
from datasets import Dataset, DatasetDict
from config import *


def get_data(book_path):
    """
    Extract data from the book files.

    Args:
        book_path (str): Path to the directory containing book files.

    Returns:
        tuple: A tuple containing cleaned English and Hebrew texts.
    """
    he_dir = os.path.join(book_path, 'Hebrew/merged.json')
    en_dir = os.path.join(book_path, 'English/merged.json')

    with open(he_dir, 'r') as f:
        he = json.load(f)

    with open(en_dir, 'r') as f:
        en = json.load(f)

    en_total = en['text']['Introduction']
    for x in en['text']['']:
        en_total += x

    he_total = he['text']['Introduction']
    for x in he['text']['']:
        he_total += x

    def clean_data(data):
        """
        Clean the given data by removing HTML tags.

        Args:
            data (str): Input data containing HTML tags.

        Returns:
            str: Cleaned data without HTML tags.
        """
        # Remove HTML tags
        data = re.sub(r'<[^>]*>', '', data)
        return data

    en_total = [clean_data(x) for x in en_total]
    he_total = [clean_data(x) for x in he_total]

    return en_total, he_total


def get_datasets():
    """
    Prepare datasets for the experiments.

    Returns:
        tuple: A tuple containing raw datasets for all four experiments.
    """

    # get data from the books
    en_total1, he_total1 = get_data(book1_path)
    en_total2, he_total2 = get_data(book2_path)

    # split into eval and train and test for experiment 1
    en_train1 = en_total1[:int(len(en_total1)*0.8)]
    he_train1 = he_total1[:int(len(he_total1)*0.8)]

    en_eval1 = en_total1[int(len(en_total1)*0.8):int(len(en_total1)*0.9)]
    he_eval1 = he_total1[int(len(he_total1)*0.8):int(len(he_total1)*0.9)]

    en_test1 = en_total1[int(len(en_total1)*0.9):]
    he_test1 = he_total1[int(len(he_total1)*0.9):]

    # split into eval and train and test for experiment 2
    en_train2 = en_total1[:int(len(en_total1)*0.8)]
    he_train2 = he_total1[:int(len(he_total1)*0.8)]

    en_eval2 = en_total1[int(len(en_total1)*0.8):]
    he_eval2 = he_total1[int(len(he_total1)*0.8):]

    en_test2 = en_total2
    he_test2 = he_total2

    # split into eval and train and test for experiment 3
    en_train3 = en_total2[:int(len(en_total2)*0.8)]
    he_train3 = he_total2[:int(len(he_total2)*0.8)]

    en_eval3 = en_total2[int(len(en_total2)*0.8):]
    he_eval3 = he_total2[int(len(he_total2)*0.8):]

    en_test3 = en_total1
    he_test3 = he_total1

    # split into eval and train and test for experiment 4
    en_train4 = en_total2[:int(len(en_total2)*0.8)]
    he_train4 = he_total2[:int(len(he_total2)*0.8)]

    en_eval4 = en_total2[int(len(en_total2)*0.8):int(len(en_total2)*0.9)]
    he_eval4 = he_total2[int(len(he_total2)*0.8):int(len(he_total2)*0.9)]

    en_test4 = en_total2[int(len(en_total2)*0.9):]
    he_test4 = he_total2[int(len(he_total2)*0.9):]
    

    # create datasets for experiment 1
    train_dataset1 = Dataset.from_dict({'translation': [{'en': en_train1[i], 'he': he_train1[i]} for i in range(len(en_train1))]})
    eval_dataset1 = Dataset.from_dict({'translation': [{'en': en_eval1[i], 'he': he_eval1[i]} for i in range(len(en_eval1))]})
    test_dataset1 = Dataset.from_dict({'translation': [{'en': en_test1[i], 'he': he_test1[i]} for i in range(len(en_test1))]})
    raw_datasets1 = DatasetDict({'train': train_dataset1, 'eval': eval_dataset1, 'test': test_dataset1})

    # create datasets for experiment 2
    train_dataset2 = Dataset.from_dict({'translation': [{'en': en_train2[i], 'he': he_train2[i]} for i in range(len(en_train2))]})
    eval_dataset2 = Dataset.from_dict({'translation': [{'en': en_eval2[i], 'he': he_eval2[i]} for i in range(len(en_eval2))]})
    test_dataset2 = Dataset.from_dict({'translation': [{'en': en_test2[i], 'he': he_test2[i]} for i in range(len(en_test2))]})
    raw_datasets2 = DatasetDict({'train': train_dataset2, 'eval': eval_dataset2, 'test': test_dataset2})

    # create datasets for experiment 3
    train_dataset3 = Dataset.from_dict({'translation': [{'en': en_train3[i], 'he': he_train3[i]} for i in range(len(en_train3))]})
    eval_dataset3 = Dataset.from_dict({'translation': [{'en': en_eval3[i], 'he': he_eval3[i]} for i in range(len(en_eval3))]})
    test_dataset3 = Dataset.from_dict({'translation': [{'en': en_test3[i], 'he': he_test3[i]} for i in range(len(en_test3))]})
    raw_datasets3 = DatasetDict({'train': train_dataset3, 'eval': eval_dataset3, 'test': test_dataset3})

    # create datasets for experiment 4
    train_dataset4 = Dataset.from_dict({'translation': [{'en': en_train4[i], 'he': he_train4[i]} for i in range(len(en_train4))]})
    eval_dataset4 = Dataset.from_dict({'translation': [{'en': en_eval4[i], 'he': he_eval4[i]} for i in range(len(en_eval4))]})
    test_dataset4 = Dataset.from_dict({'translation': [{'en': en_test4[i], 'he': he_test4[i]} for i in range(len(en_test4))]})
    raw_datasets4 = DatasetDict({'train': train_dataset4, 'eval': eval_dataset4, 'test': test_dataset4})

    return raw_datasets1, raw_datasets2, raw_datasets3, raw_datasets4
