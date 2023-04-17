"""Replay parser
"""


import sqlite3
from sqlite3.dbapi2 import Connection, Error
from typing import List

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from .replay_attack_record import ReplayAttackRecord


class ReplayAttackParser:
    """Replay attack parser"""

    def __init__(self) -> None:
        load_dotenv()

    @staticmethod
    def parse_list_file(path) -> List[ReplayAttackRecord]:
        """Parse list

        Args:
            path (str): The list path

        Returns:
            List[ReplayAttackRecord]: records
        """
        with open(path, "r") as f:
            lines = f.readlines()

        ret = []

        for line in lines:
            line = line.rstrip()
            split = line.split("/")
            dataset = split[1]
            name = split[-1]
            client = [s for s in name.split("_") if "client" in s]
            client = client[0] if len(client) >= 1 else ""
            genuine = "real"

            if "attack" in line:
                genuine = "attack"

            record = ReplayAttackRecord(line, name, dataset, genuine, client)

            ret.append(record)

        return ret

    @staticmethod
    def create_connection(db_file) -> Connection:
        """create a database connection to the SQLite database
            specified by db_file
        :param db_file: database file
        :return: Connection object or None
        """
        conn = None
        try:
            conn = sqlite3.connect(db_file)
        except Error as exception:
            print(exception)

        return conn

    @staticmethod
    def insert_db(records: List[ReplayAttackRecord], path: str):
        """Inserts records in db

        Args:
            records (List[ReplayAttackRecord]): Replay attack records
        """
        conn = ReplayAttackParser.create_connection(path)
        with conn:
            cur = conn.cursor()
            cur.execute("DELETE FROM replay_attack_record")
            cur.executemany(
                "INSERT INTO replay_attack_record(path, filename, dataset, genuine, client) \
                    values (?, ?, ?, ?, ?)",
                (
                    (obj.path, obj.filename, obj.dataset, obj.genuine, obj.client)
                    for obj in records
                ),
            )

    @staticmethod
    def get_db_records_as_frame(datasets: List[str], sqlite3_path: str) -> pd.DataFrame:
        """Returns the info in the DB

        Returns:
            pd.DataFrame: pd.Dataframe of ReplayAttackRecord
        """
        with ReplayAttackParser.create_connection(sqlite3_path) as conn:
            datasets_str = ", ".join(f'"{w}"' for w in datasets)
            sql = (
                f"select * from replay_attack_record where dataset in ({datasets_str})"
            )
            ret = pd.read_sql_query(sql, conn)

        ret["label"] = np.where(ret["genuine"] == "real", 0, 1)

        return ret[["path", "label", "client"]]
