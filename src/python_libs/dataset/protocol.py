"""Protocol
"""

import json
import os
from typing import List

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

        records = []
        records.extend(Protocol.load_rose(json_path))
        print("Rose loaded...")
        records.extend(Protocol.load_replay_mobile(json_path))
        print("Replay loaded...")
        return train_test_split(records, test_size=0.3)

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
