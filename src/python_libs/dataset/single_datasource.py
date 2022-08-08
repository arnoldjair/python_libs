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

    def __init__(self, dataset, time=300, width=416, height=416):
        super().__init__()
        self.dataset = dataset
        self.time = time
        self.width = width
        self.height = height

    def __getitem__(self, index):
        record_0 = self.dataset[index]
        label = record_0.label

        return (
            torch.from_numpy(
                record_0.load(
                    time=300, scale=True, width=self.width, height=self.height
                )
            ).float(),
            label,
        )

    def __len__(self):
        return len(self.dataset)
