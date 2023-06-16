"""
DataSource
"""
import numpy as np
import torch
from torch.utils.data import Dataset


class TripletEvalDatasource(Dataset):
    def __init__(
        self,
        evaluation_dataset,
        training_dataset,
        time=300,
        width=416,
        height=416,
        samples=30,
    ):
        """_summary_

        Args:
            evaluation_dataset (_type_): _description_
            training_dataset (_type_): _description_
            time (int, optional): _description_. Defaults to 300.
            width (int, optional): _description_. Defaults to 416.
            height (int, optional): _description_. Defaults to 416.
            samples (int, optional): _description_. Defaults to 30.
        """
        super().__init__()
        self.evaluation_dataset = evaluation_dataset
        self.training_dataset = training_dataset
        self.time = time
        self.width = width
        self.height = height
        self.samples = samples

    def __getitem__(self, index):
        record_0 = self.evaluation_dataset[index]

        if index % 2 == 0:
            same_user_records = [
                curr
                for curr in self.training_dataset
                if curr.label == record_0.label
                and curr.dataset == record_0.dataset
                and curr.client == record_0.client
            ]
        else:
            same_user_records = [
                curr
                for curr in self.training_dataset
                if curr.label != record_0.label
                and curr.dataset == record_0.dataset
                and curr.client == record_0.client
            ]

        if len(same_user_records) == 0:
            same_user_records = self.training_dataset

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
        return len(self.evaluation_dataset)
