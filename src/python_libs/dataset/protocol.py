"""Protocol
"""

import json
import os
from typing import List

from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

from .record import Record
from .rose.rose_parser import RoseParser


class Protocol:  # pylint: disable=too-few-public-methods
    """Class for handling protocol"""

    def __init__(self) -> None:
        load_dotenv()

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

            rose_path = os.environ.get("ROSE_DATASET")
            print(rose_path)
            subjects = json_file["rose"]["subjects"]
            frames = RoseParser.get_db_records_as_frame(subjects)
            records = list(
                map(
                    lambda record: Record(
                        os.path.join(rose_path, record[0]), record[1]
                    ),
                    frames.to_numpy(),
                )
            )

        return train_test_split(records, test_size=0.3)
