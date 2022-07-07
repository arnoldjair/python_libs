"""
DataSource
"""
import random

import torch
from torch.utils.data import Dataset


class Datasource(Dataset):
    """Datasource

    Args:
        Dataset (Records): Datasource
    """

    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def __getitem__(self, index):
        # We need approx 50 % of  records of the same class
        same_class = random.randint(0, 1)
        record_0 = self.dataset[index]
        label_0 = record_0.label
        label = 0
        if same_class:
            while True:
                # keep looping till the same class record is found
                index_1 = random.randint(0, self.__len__() - 1)
                record_1 = self.dataset[index_1]
                label_1 = record_1.label
                if label_0 == label_1:
                    label = 1
                    break
        else:
            while True:
                index_1 = random.randint(0, self.__len__() - 1)
                record_1 = self.dataset[index_1]
                label_1 = record_1.label
                if label_0 != label_1:
                    break

        return (
            torch.from_numpy(record_0.load(time=300)).float(),
            torch.from_numpy(record_1.load(time=300)).float(),
            label,
        )

    def __len__(self):
        return len(self.dataset)
