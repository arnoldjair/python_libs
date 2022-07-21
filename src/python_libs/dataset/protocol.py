"""Protocol
"""

import json
import os
from typing import List

from sklearn.model_selection import train_test_split

from .record import Record
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

        return train_test_split(records, test_size=0.3)
