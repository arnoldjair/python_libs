"""
DataSource
"""
import numpy as np
import torch
from torch.utils.data import Dataset


class Datasource(Dataset):
    """Datasource

    Args:
        Dataset (Records): Datasource
    """

    def __init__(self, dataset, time=300, width=416, height=416, samples=30):
        super().__init__()
        self.dataset = dataset
        self.time = time
        self.width = width
        self.height = height
        self.samples = samples

    def __getitem__(self, index):
        record_0 = self.dataset[index]

        if index % 2 == 0:
            records = [curr for curr in self.dataset if curr.label == record_0.label]
        else:
            records = [curr for curr in self.dataset if curr.label != record_0.label]

        record_1 = np.random.choice(records)

        return (
            torch.from_numpy(
                record_0.load(time=self.time, samples=self.samples)
            ).float(),
            torch.from_numpy(
                record_1.load(time=self.time, samples=self.samples)
            ).float(),
            0 if record_0.label == record_1.label else 1,
        )

    def __len__(self):
        return len(self.dataset)
