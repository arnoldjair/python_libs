"""Protocol
"""

import json
import logging
import os
from typing import List

import numpy as np
from sklearn.model_selection import train_test_split

from .record import Record
from .replay_mobile.replay_parser import ReplayParser
from .rose.rose_parser import RoseParser


class Protocol:  # pylint: disable=too-few-public-methods
    """Class for handling protocol"""

    def __init__(self) -> None:
        pass

    @staticmethod
    def get_protocol(json_path: str) -> List[Record]:
        """Get protocol defined in json file

        Args:
            path (str): Json file path

        Returns:
            List[Record]: List of Records
        """
        logger = logging.getLogger("antispoofing.protocol")
        records = []
        records.extend(Protocol.load_rose(json_path))
        logger.info("Rose loaded...")
        records.extend(Protocol.load_replay_mobile(json_path))
        logger.info("Replay loaded...")
        return train_test_split(records, test_size=0.3)

    @staticmethod
    def get_pairs_protocol(json_path: str):
        """Get pairs protocol defined in json file

        Args:
             path (str): Json file path

        Returns:
            List[Record, Record, int]: List of pairs of records with label (1 is equals, 0 if not)
        """

        logger = logging.getLogger("antispoofing.pairs_protocol")
        records = []
        pairs_train = []
        pairs_test = []
        records.extend(Protocol.load_rose(json_path))
        logger.info("Rose loaded...")
        records.extend(Protocol.load_replay_mobile(json_path))
        logger.info("Replay loaded...")

        train, test = train_test_split(records, test_size=0.3)

        fraud_train = [curr for curr in train if curr.label == 1]
        genuine_train = [curr for curr in train if curr.label == 0]

        fraud_test = [curr for curr in test if curr.label == 1]
        genuine_test = [curr for curr in test if curr.label == 0]

        for record in train:
            pairs_train.append(
                [record, np.random.choice(genuine_train), 1 if record.label == 0 else 0]
            )
            pairs_train.append(
                [record, np.random.choice(fraud_train), 1 if record.label == 1 else 0]
            )

        for record in test:
            pairs_test.append(
                [record, np.random.choice(genuine_test), 1 if record.label == 0 else 0]
            )
            pairs_test.append(
                [record, np.random.choice(fraud_test), 1 if record.label == 1 else 0]
            )

        return [pairs_train, pairs_test]

    @staticmethod
    def load_rose(json_path: str):
        with open(json_path, "r") as file:
            json_file = json.load(file)

            rose_path = json_file["rose"]["root_path"]
            subjects = json_file["rose"]["subjects"]
            sqlite3_path = json_file["rose"]["sqlite3_path"]
            frames = RoseParser.get_db_records_as_frame(
                ids=subjects, sqlite3_path=sqlite3_path
            )
            records = list(
                map(
                    lambda record: Record(
                        os.path.join(rose_path, record[0]), record[1]
                    ),
                    frames.to_numpy(),
                )
            )

        return records

    @staticmethod
    def load_replay_mobile(json_path: str):
        with open(json_path, "r") as file:
            json_file = json.load(file)
            replay_mobile_path = json_file["replay_mobile"]["root_path"]
            datasets = json_file["replay_mobile"]["datasets"]
            sqlite3_path = json_file["replay_mobile"]["sqlite3_path"]
            frames = ReplayParser.get_db_records_as_frame(
                datasets=datasets, sqlite3_path=sqlite3_path
            )
            records = list(
                map(
                    lambda record: Record(
                        os.path.join(replay_mobile_path, record[0]), record[1]
                    ),
                    frames.to_numpy(),
                )
            )

        return records
