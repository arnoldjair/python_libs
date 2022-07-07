"""
DataSource
"""
import torch
from torch.utils.data import Dataset


class SingleDatasource(Dataset):
    """SingleDatasource

    Args:
        Dataset (Records): Datasource
    """

    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def __getitem__(self, index):
        record_0 = self.dataset[index]
        label = 0

        return torch.from_numpy(record_0.load(time=300)).float(), label

    def __len__(self):
        return len(self.dataset)
