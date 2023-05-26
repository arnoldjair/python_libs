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

    def __init__(self, dataset, time=300, width=416, height=416, samples=30):
        super().__init__()
        self.dataset = dataset
        self.time = time
        self.width = width
        self.height = height
        self.samples = samples

    def __getitem__(self, index):
        record_0, record_1, label = self.dataset[index]

        return (
            torch.from_numpy(
                record_0.load(time=self.time, samples=self.samples)
            ).float(),
            torch.from_numpy(
                record_1.load(time=self.time, samples=self.samples)
            ).float(),
            label,
        )

    def __len__(self):
        return len(self.dataset)
