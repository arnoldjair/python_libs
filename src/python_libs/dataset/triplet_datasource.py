"""
DataSource
"""
import numpy as np
import torch
from torch.utils.data import Dataset


class TripletDatasource(Dataset):
    def __init__(self, dataset, time=300, width=416, height=416, samples=30):
        """_summary_

        Args:
            dataset (_type_): _description_
            time (int, optional): _description_. Defaults to 300.
            width (int, optional): _description_. Defaults to 416.
            height (int, optional): _description_. Defaults to 416.
            samples (int, optional): _description_. Defaults to 30.
        """
        super().__init__()
        self.dataset = dataset
        self.time = time
        self.width = width
        self.height = height
        self.samples = samples

    def __getitem__(self, index):
        record_0 = self.dataset[index]

        if index % 2 == 0:
            same_user_records = [
                curr
                for curr in self.dataset
                if curr.label == record_0.label
                and curr.dataset == record_0.dataset
                and curr.client == record_0.client
            ]
        else:
            same_user_records = [
                curr
                for curr in self.dataset
                if curr.label != record_0.label
                and curr.dataset == record_0.dataset
                and curr.client == record_0.client
            ]

        record_1 = np.random.choice(same_user_records)

        return (
            torch.from_numpy(
                record_0.load(time=self.time, samples=self.samples)
            ).float(),
            torch.from_numpy(
                record_1.load(time=self.time, samples=self.samples)
            ).float(),
            1 if record_0.label == record_1.label else 0,
        )

    def __len__(self):
        return len(self.dataset)
