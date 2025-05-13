#!/usr/bin/env python3

import sqlite3
import pickle
import os
from datasets import load_dataset
from typing import Set, Any, List, Tuple

from . import dataset_writer

SQLITE_BATCH = 30


def iter_dataset(ds):
    i = 0
    for d in ds:
        yield (i, d)
        i += 1


def select_rows_excluding_set(
    cursor: sqlite3.Cursor,
    table_name: str,
    column_name: str,
    exclude_values: Set[Any],
    select_columns: str = "*",
) -> sqlite3.Cursor:
    """
    Selects rows from `table_name` where `column_name` is NOT in `exclude_values`.
    Uses a temporary table to avoid SQLite parameter limits.

    Args:
        cur: sqlite3.Cursor object
        table_name: Name of the main table
        column_name: Column to apply the exclusion filter
        exclude_values: Set of values to exclude
        select_columns: Columns to select (default is '*')

    Returns:
        List of matching rows as tuples.
    """

    if not exclude_values:
        # If the set is empty, return all rows
        query = f"SELECT {select_columns} FROM {table_name}"
        cursor.execute(query)
        return cursor

    # Create a temporary table for exclusion
    cursor.execute("DROP TABLE IF EXISTS temp_exclude")
    cursor.execute(f"CREATE TEMP TABLE temp_exclude ({column_name} TEXT PRIMARY KEY)")

    # Bulk insert into temp table
    cursor.executemany(
        f"INSERT INTO temp_exclude ({column_name}) VALUES (?)", [(val,) for val in exclude_values]
    )

    # Query excluding matching entries
    query = f"""
        SELECT {select_columns} FROM {table_name}
        WHERE {column_name} NOT IN (
            SELECT {column_name} FROM temp_exclude
        )
    """
    cursor.execute(query)
    return cursor


class SqliteDatasetReader:
    def __init__(self, path, already_processed=None):
        self.ty = "sqlite3"
        self.con = sqlite3.connect(path)
        self.con.row_factory = sqlite3.Row
        self.cur = self.con.cursor()
        self.blob_storage_path = path + ".storage"
        if already_processed is None:
            self.already_processed = {}
        else:
            self.already_processed = already_processed

    def cleanup(self):
        self.con.close()

    def iter_sqlite(self):
        while True:
            rows = list(self.cur.fetchmany(SQLITE_BATCH))
            if not rows:
                break
            for d in rows:
                new_d = {}
                for k, v in dict(d).items():
                    if isinstance(v, bytes):
                        if v[0] == 0:
                            new_v = v[1:]
                        else:
                            new_v = self.load_blob_from_file(v[1:].decode("utf-8"))
                    else:
                        new_v = v
                    if k.startswith(dataset_writer.PICKLED_COLUMN_PREFIX):
                        new_k = k.removeprefix(dataset_writer.PICKLED_COLUMN_PREFIX)
                        new_v = pickle.loads(new_v)
                    else:
                        new_k = k
                    new_d[new_k] = new_v
                rowid = new_d["rowid"]
                del new_d["rowid"]
                yield (rowid, new_d)

    def get_iter(self):
        self.cur.execute("SELECT rowid, * FROM data")
        select_rows_excluding_set(
            self.cur,
            dataset_writer.DATA_TABLE,
            dataset_writer.PRIMARY_KEY,
            self.already_processed,
            "rowid, *",
        )

        return self.iter_sqlite()

    def load_blob_from_file(self, name):
        with open(os.path.join(self.blob_storage_path, name), "rb") as f:
            return f.read()

    def get_one_iter(self, one):
        self.cur.execute("SELECT rowid, * FROM data WHERE rowid=?", (one,))
        return self.iter_sqlite()


class DatasetsDatasetReader:
    def __init__(self, path):
        self.ty = "ds"
        self.ds = load_dataset(path, split="train", streaming=True)

    def cleanup(self): ...

    def get_iter(self):
        return iter_dataset(self.ds)

    def get_one_iter(self, one):
        return iter_dataset(self.ds.skip(one))


class DatasetReader:
    def __init__(self, path, already_processed=None):
        self.path = path
        self.already_processed = already_processed

    def __enter__(self):
        self.impl = None
        if os.path.isfile(self.path):
            self.impl = SqliteDatasetReader(self.path, self.already_processed)
        elif os.path.isdir(self.path):
            self.impl = DatasetsDatasetReader(self.path)
        else:
            raise FileNotFoundError(self.path)
        return self.impl

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.impl.cleanup()
