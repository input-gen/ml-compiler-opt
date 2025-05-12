#!/usr/bin/env python3

import sqlite3
import pickle
import os

from datasets import load_dataset
from . import dataset_writer


def iter_dataset(ds):
    i = 0
    for d in ds:
        yield (i, d)
        i += 1


SQLITE_BATCH = 30


class SqliteDatasetReader:
    def __init__(self, path):
        self.ty = "sqlite3"
        self.con = sqlite3.connect(path)
        self.con.row_factory = sqlite3.Row
        self.cur = self.con.cursor()
        self.blob_storage_path = path + ".storage"

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
                yield (new_d["id"], new_d)

    def get_iter(self):
        self.cur.execute("SELECT * FROM data")
        return self.iter_sqlite()

    def load_blob_from_file(self, name):
        with open(os.path.join(self.blob_storage_path, name), "rb") as f:
            return f.read()

    def get_one_iter(self, one):
        self.cur.execute("SELECT * FROM data WHERE rowid=?", (one,))
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
