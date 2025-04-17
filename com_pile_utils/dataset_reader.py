#!/usr/bin/env python3

import sqlite3
import os

from datasets import load_dataset


def iter_dataset(ds):
    i = 0
    for d in ds:
        yield (i, d)
        i += 1


class DatasetReader:
    def __init__(self, path):
        if os.path.isfile(path):
            self.ty = "sqlite3"
            self.con = sqlite3.connect(path)
            self.cur = self.con.cursor()
        elif os.path.isdir(path):
            self.ty = "ds"
            self.ds = load_dataset(path, split="train", streaming=True)
        else:
            raise FileNotFoundError(path)

    def __del__(self):
        self.con.close()

    def get_iter(self):
        if self.ty == "sqlite3":
            return self.cur.execute(f"SELECT rowid, data FROM data")
        elif self.ty == "ds":
            return iter_dataset(self.ds)
        else:
            raise Exception("unknown type")
