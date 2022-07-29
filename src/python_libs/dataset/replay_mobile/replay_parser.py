"""Replay parser
"""


import sqlite3
from sqlite3.dbapi2 import Connection, Error
from typing import List

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from .replay_record import ReplayRecord


class ReplayParser:
    """Replay parser"""

    def __init__(self) -> None:
        load_dotenv()

    @staticmethod
    def parse_list_file(path) -> List[ReplayRecord]:
        """Parse list

        Args:
            path (str): The list path

        Returns:
            List[ReplayRecord]: records
        """
        with open(path, "r") as f:
            lines = f.readlines()

        ret = []

        for line in lines:
            line = line.rstrip()
            fields = line.split("/")[-1].split(".")[0].split("_")
            dataset = line.split("/")[1]
            if len(fields) == 8:
                (
                    genuine,
                    client,
                    session,
                    attack,
                    support,
                    device,
                    presentation,
                    light,
                ) = fields
                record = ReplayRecord(
                    line,
                    line.split("/")[-1],
                    dataset,
                    genuine,
                    client,
                    session,
                    attack,
                    support,
                    device,
                    presentation,
                    light,
                    None,
                    None,
                    None,
                )
            else:
                client, session, step, device, light_condition = fields
                record = ReplayRecord(
                    line,
                    line.split("/")[-1],
                    dataset,
                    "real",
                    client,
                    session,
                    None,
                    None,
                    device,
                    None,
                    None,
                    step,
                    light_condition,
                    None,
                )
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
    def insert_db(records: List[ReplayRecord], path: str):
        """Inserts records in db

        Args:
            records (List[ReplayRecord]): Replay records
        """
        conn = ReplayParser.create_connection(path)
        with conn:
            cur = conn.cursor()
            cur.execute("DELETE FROM replay_record")
            cur.executemany(
                "INSERT INTO replay_record(path, filename, dataset, genuine, client, session, attack, support, device, presentation, light, step, light_condition) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    (
                        obj.path,
                        obj.filename,
                        obj.dataset,
                        obj.genuine,
                        obj.client,
                        obj.session,
                        obj.attack,
                        obj.support,
                        obj.device,
                        obj.presentation,
                        obj.light,
                        obj.step,
                        obj.light_condition,
                    )
                    for obj in records
                ),
            )

    @staticmethod
    def get_db_records_as_frame(datasets: List[str], sqlite3_path: str) -> pd.DataFrame:
        """Returns the info in the DB

        Returns:
            pd.DataFrame: pd.Dataframe of ReplayRecord
        """
        with ReplayParser.create_connection(sqlite3_path) as conn:
            datasets_str = ", ".join(f'"{w}"' for w in datasets)
            sql = f"select * from replay_record where dataset in ({datasets_str})"
            ret = pd.read_sql_query(sql, conn)

        ret = ret.sort_values(by=["client"], ascending=True)
        ret["label"] = np.where(ret["genuine"] == "real", 0, 1)

        return ret[["path", "label"]]
