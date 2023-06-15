"""Protocol
"""

import json
import logging
import os
import random
from typing import List

import numpy as np

from .record import Record
from .replay_attack.replay_attack_parser import ReplayAttackParser
from .replay_mobile.replay_parser import ReplayParser
from .rose.rose_parser import RoseParser


class Protocol:  # pylint: disable=too-few-public-methods
    """Class for handling protocol"""

    def __init__(self) -> None:
        pass

    @staticmethod
    def get_records(json_path: str) -> List[Record]:
        """Get records defined in json file

        Args:
            path (str): Json file path

        Returns:
            List[Record], Dict: List of Records and groups
        """
        logger = logging.getLogger("antispoofing.records")
        records = []
        groups = {}

        with open(json_path, "r") as file:
            json_file = json.load(file)

            if "rose" in json_file:
                rose_records, rose_groups = Protocol.load_rose(json_path)
                records.extend(rose_records)
                groups["rose"] = rose_groups
                logger.info("Rose loaded...")

            if "replay_mobile" in json_file:
                (
                    replay_mobile_records,
                    replay_mobile_groups,
                ) = Protocol.load_replay_mobile(json_path)
                records.extend(replay_mobile_records)
                groups["replay_mobile"] = replay_mobile_groups
                logger.info("Replay mobile loaded...")

            if "replay_attack" in json_file:
                (
                    replay_attack_records,
                    replay_attack_groups,
                ) = Protocol.load_replay_attack(json_path)
                records.extend(replay_attack_records)
                groups["replay_attack"] = replay_attack_groups
                logger.info("Replay attack loaded...")

        return records, groups

    @staticmethod
    def get_validation_records(validation_json_path: str):

        logger = logging.getLogger("antispoofing.get_validation_records")

        records_validation_rose = []
        records_validation_replay_attack = []
        records_validation_replay_mobile = []

        with open(validation_json_path, "r") as file:
            json_file = json.load(file)

            if "rose" in json_file:
                # records.extend(Protocol.load_rose(json_path))
                records_validation_rose = Protocol.load_rose(validation_json_path)
                logger.info("Rose loaded...")

            if "replay_mobile" in json_file:
                records_validation_replay_mobile = Protocol.load_replay_mobile(
                    validation_json_path
                )
                logger.info("Replay mobile loaded...")

            if "replay_attack" in json_file:
                records_validation_replay_attack = Protocol.load_replay_attack(
                    validation_json_path
                )
                logger.info("Replay attack loaded...")

        return {
            "rose": records_validation_rose,
            "replay_attack": records_validation_replay_attack,
            "replay_mobile": records_validation_replay_mobile,
        }

    @staticmethod
    def get_training_records(training_json_path: str):

        logger = logging.getLogger("antispoofing.get_training_records")

        records_training_rose = []
        records_training_replay_attack = []
        records_training_replay_mobile = []

        with open(training_json_path, "r") as file:
            json_file = json.load(file)

            if "rose" in json_file:
                # records.extend(Protocol.load_rose(json_path))
                records_training_rose = Protocol.load_rose(training_json_path)
                logger.info("Rose loaded...")

            if "replay_mobile" in json_file:
                records_training_replay_mobile = Protocol.load_replay_mobile(
                    training_json_path
                )
                logger.info("Replay mobile loaded...")

            if "replay_attack" in json_file:
                records_training_replay_attack = Protocol.load_replay_attack(
                    training_json_path
                )
                logger.info("Replay attack loaded...")

        return {
            "rose": records_training_rose,
            "replay_attack": records_training_replay_attack,
            "replay_mobile": records_training_replay_mobile,
        }

    @staticmethod
    def get_validation_protocol(validation_json_path: str, training_json_path: str):

        ret = {"rose": [], "replay_attack": [], "replay_mobile": []}

        validation_records = Protocol.get_validation_records(validation_json_path)
        training_dataset_records = Protocol.get_training_records(training_json_path)

        training_records = []
        training_records.extend(training_dataset_records["rose"])
        training_records.extend(training_dataset_records["replay_mobile"])
        training_records.extend(training_dataset_records["replay_attack"])

        ret["rose"] = Protocol.pair_records(
            validation_records["rose"], training_records
        )
        ret["replay_attack"] = Protocol.pair_records(
            validation_records["replay_attack"], training_records
        )
        ret["replay_mobile"] = Protocol.pair_records(
            validation_records["replay_mobile"], training_records
        )

        return ret

    @staticmethod
    def pair_records(records, train_records):

        ret = []

        for record in records:
            random_record = np.random.choice(train_records)
            ret.append(
                [
                    record,
                    random_record,
                    1 if record.label == random_record.label else 0,
                ]
            )

        return ret

    @staticmethod
    def get_pairs_protocol(json_path: str, same_user=True):
        """Get pairs protocol defined in json file

        Args:
             path (str): Json file path

        Returns:
            List[Record, Record, int]: List of pairs of records with label (1 is equals, 0 if not)
        """

        logger = logging.getLogger("antispoofing.pairs_protocol")
        records = []
        records_rose = []
        records_replay_attack = []
        records_replay_mobile = []
        pairs_train = []

        with open(json_path, "r") as file:
            json_file = json.load(file)

            if "rose" in json_file:
                # records.extend(Protocol.load_rose(json_path))
                records_rose = Protocol.load_rose(json_path)
                records.extend(records_rose)
                logger.info("Rose loaded...")

            if "replay_mobile" in json_file:
                records_replay_mobile = Protocol.load_replay_mobile(json_path)
                records.extend(records_replay_mobile)
                logger.info("Replay mobile loaded...")

            if "replay_attack" in json_file:
                records_replay_attack = Protocol.load_replay_attack(json_path)
                records.extend(records_replay_attack)
                logger.info("Replay attack loaded...")

        random.shuffle(records)

        fraud_train = [curr for curr in records if curr.label == 1]
        genuine_train = [curr for curr in records if curr.label == 0]

        for index, record in enumerate(records):
            if index % 2 == 0 and same_user:
                # list comprehension where user and dataset
                genuine_user_records = [
                    curr
                    for curr in records
                    if curr.dataset == record.dataset
                    and curr.client == record.client
                    and curr.label == 0
                ]

                if len(genuine_user_records) == 0:
                    genuine_user_records = genuine_train

                fraud_user_records = [
                    curr
                    for curr in records
                    if curr.dataset == record.dataset
                    and curr.client == record.client
                    and curr.label == 1
                ]

                if len(fraud_user_records) == 0:
                    fraud_user_records = fraud_train

                pairs_train.append(
                    [
                        record,
                        np.random.choice(genuine_user_records),
                        1 if record.label == 0 else 0,
                    ]
                )
                pairs_train.append(
                    [
                        record,
                        np.random.choice(fraud_user_records),
                        1 if record.label == 1 else 0,
                    ]
                )
            else:
                pairs_train.append(
                    [
                        record,
                        np.random.choice(genuine_train),
                        1 if record.label == 0 else 0,
                    ]
                )
                pairs_train.append(
                    [
                        record,
                        np.random.choice(fraud_train),
                        1 if record.label == 1 else 0,
                    ]
                )

        fraud_train = [curr for curr in records if curr.label == 1]
        genuine_train = [curr for curr in records if curr.label == 0]

        return [pairs_train, []]

    @staticmethod
    def group_by_client(records: List[Record], client_ids: List[str]):
        ret = {}
        for curr_id in client_ids:
            # We're assuming that all records belong to the same dataset
            client_records = [curr for curr in records if curr.client == curr_id]
            ret[curr_id] = client_records

        return ret

    @staticmethod
    def load_rose(json_path: str, group_by_client=False):
        groups = {}
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
                        os.path.join(rose_path, record[0]), record[1], "rose", record[2]
                    ),
                    frames.to_numpy(),
                )
            )

        if group_by_client:
            groups = Protocol.group_by_client(records, subjects)

        return records, groups

    @staticmethod
    def load_replay_mobile(json_path: str, group_by_client=False):
        groups = {}
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
                        os.path.join(replay_mobile_path, record[0]),
                        record[1],
                        "replay_mobile",
                        record[2],
                    ),
                    frames.to_numpy(),
                )
            )

        if group_by_client:
            groups = Protocol.group_by_client(records, datasets)

        return records, groups

    @staticmethod
    def load_replay_attack(json_path: str, group_by_client=False):
        groups = {}
        with open(json_path, "r") as file:
            json_file = json.load(file)
            replay_attack_path = json_file["replay_attack"]["root_path"]
            datasets = json_file["replay_attack"]["datasets"]
            sqlite3_path = json_file["replay_attack"]["sqlite3_path"]
            frames = ReplayAttackParser.get_db_records_as_frame(
                datasets=datasets, sqlite3_path=sqlite3_path
            )
            records = list(
                map(
                    lambda record: Record(
                        os.path.join(replay_attack_path, record[0]),
                        record[1],
                        "replay_attack",
                        record[2],
                    ),
                    frames.to_numpy(),
                )
            )

        if group_by_client:
            groups = Protocol.group_by_client(records, datasets)

        return records, groups
