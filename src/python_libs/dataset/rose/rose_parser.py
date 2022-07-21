"""Rose parser
"""


import os
import sqlite3
from sqlite3.dbapi2 import Connection, Error
from typing import List

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from .rose_record import RoseRecord


class RoseParser:
    """Rose parser"""

    def __init__(self) -> None:
        load_dotenv()
        pass

    @staticmethod
    def parse_list_file(path) -> List[RoseRecord]:
        """Parse list

        Args:
            path (str): The list path

        Returns:
            List[RoseRecord]: records
        """
        with open(path, "r") as f:
            lines = f.readlines()

        ret = []

        for line in lines:
            line = line.rstrip()
            l, s, d, x, e, p, n = line.split("/")[-1].split(".")[0].split("_")
            record = RoseRecord(line, line.split("/")[-1], l, s, d, x, e, p, n, None)
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
    def insert_db(records: List[RoseRecord]):
        """Inserts records in db

        Args:
            records (List[RoseRecord]): Rose records
        """
        conn = RoseParser.create_connection(
            os.path.join(os.environ.get("ROSE_DATASET"), "rose.sqlite3")
        )
        with conn:
            cur = conn.cursor()
            cur.execute("DELETE FROM rose_record")
            cur.executemany(
                "INSERT INTO rose_record(path, filename, l, s, d, x, e, p, n) values (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    (
                        obj.path,
                        obj.filename,
                        obj.l,
                        obj.s,
                        obj.d,
                        obj.x,
                        obj.e,
                        obj.p,
                        obj.n,
                    )
                    for obj in records
                ),
            )

    @staticmethod
    def get_db_records_as_frame(ids: List[str], sqlite3_path: str) -> pd.DataFrame:
        """Returns the info in the DB

        Returns:
            pd.DataFrame: pd.Dataframe of RoseRecord
        """
        with RoseParser.create_connection(sqlite3_path) as conn:
            ret = pd.read_sql_query(
                f'select * from rose_record where p in ({",".join(ids)})', conn
            )
        ret = ret.astype({"p": "int32", "n": "int32"})

        ret = ret.sort_values(by=["n"], ascending=True)
        ret["label"] = np.where(ret["l"] == "G", 0, 1)

        return ret[["path", "label"]]
