"""
DataSource
"""
import torch
from torch.utils.data import Dataset


class Datasource(Dataset):
    """Datasource

    Args:
        Dataset (Records): Datasource
    """

    def __init__(self, dataset, time=300, width=416, height=416):
        super().__init__()
        self.dataset = dataset

    def __getitem__(self, index):
        record_0, record_1, label = self.dataset[index]

        return (
            torch.from_numpy(record_0.load(time=300)).float(),
            torch.from_numpy(record_1.load(time=300)).float(),
            label,
        )

    def __len__(self):
        return len(self.dataset)
