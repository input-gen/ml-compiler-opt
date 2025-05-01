#!/usr/bin/env python3

import sqlite3
import pickle
import os

from datasets import load_dataset


def iter_dataset(ds):
    i = 0
    for d in ds:
        yield (i, d)
        i += 1


def iter_sqlite(ds):
    for i, d in ds:
        yield (i, pickle.loads(d))


class SqliteDatasetReader:
    def __init__(self, path):
        self.ty = "sqlite3"
        self.con = sqlite3.connect(path)
        self.cur = self.con.cursor()

    def cleanup(self):
        self.con.close()

    def get_iter(self):
        return iter_sqlite(self.cur.execute("SELECT rowid, data FROM data"))

    def get_iter_unprocessed(self):
        return iter_sqlite(
            self.cur.execute(
                """
            SELECT rowid, data
            FROM data WHERE id NOT IN (
              SELECT id
              FROM processed
            )
            """
            )
        )

    def get_one_iter(self, one):
        return iter_sqlite(self.cur.execute("SELECT rowid, data FROM data WHERE rowid=?", (one,)))


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
    def __init__(self, path):
        self.path = path

    def __enter__(self):
        self.impl = None
        if os.path.isfile(self.path):
            self.impl = SqliteDatasetReader(self.path)
        elif os.path.isdir(self.path):
            self.impl = DatasetsDatasetReader(self.path)
        else:
            raise FileNotFoundError(self.path)
        return self.impl

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.impl.cleanup()
