#!/usr/bin/env python3

import glob
import re
import ray
import argparse
import json
import signal
import pandas
import pyarrow
import os
import dataclasses
import logging
import sqlite3
import pickle

# from absl import flags

from typing import Optional, List
from pyarrow import parquet
from datasets import load_dataset

# flags.DEFINE_string("dataset", None)
# FLAGS = flags.FLAGS

logger = logging.getLogger(__name__)

# 100MB
PARQUET_SIZE = 100 * 1000 * 1000
VER = "1"


@dataclasses.dataclass(frozen=True)
class ProcessResult:
    i: int
    data: Optional[List[dict]]


class VersionMismatch(Exception):
    expected: str
    found: str
    pass


class CorruptMetadata(Exception):
    pass


DATA_TABLE = "data"
PROCESSED_TABLE = "processed"

ID_FIELD = "id"
SUCCESS_FIELD = "success"


def infer_sqlite_type(value):
    if isinstance(value, int):
        return "INTEGER"
    elif isinstance(value, str):
        return "TEXT"
    elif isinstance(value, bytes):
        return "BLOB"
    else:
        return None


class DatasetWriter:
    def __init__(self, output_dataset):
        self.output_dataset = output_dataset
        self.con = sqlite3.connect(self.output_dataset, timeout=60)
        self.setup_database()

        self.should_break = False
        self.should_break_immediately = False

        self.i = 0

        self.cols_added = False
        self.data_schema = None

        signal.signal(signal.SIGUSR2, self.receive)
        signal.signal(signal.SIGUSR1, self.receive)
        signal.signal(signal.SIGINT, self.receive)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.con.close()

    def receive(self, signum, stack):
        if signum == signal.SIGUSR1:
            print(f"Progress: module {self.i}")
        elif signum == signal.SIGUSR2:
            print("Will break after all pending are done.")
            self.should_break = True
        elif signum == signal.SIGINT:
            print("Will break immediately, please wait.")
            self.should_break = True
            self.should_break_immediately = True

    def setup_database(self):
        self.cur = self.con.cursor()
        self.cur.execute(f"CREATE TABLE IF NOT EXISTS {DATA_TABLE}({ID_FIELD})")
        self.cur.execute(f"CREATE TABLE IF NOT EXISTS {PROCESSED_TABLE}({ID_FIELD}, {SUCCESS_FIELD})")
        self.con.commit()

        self.already_processed = {
            r for (r,) in self.cur.execute(f"SELECT {ID_FIELD} FROM {PROCESSED_TABLE}").fetchall()
        }
        logger.debug(f"Already processed {self.already_processed}")

    def add_failure(self, idx):
        self.cur.execute(
            f"INSERT INTO {PROCESSED_TABLE} ({ID_FIELD}, {SUCCESS_FIELD}) VALUES(?, ?)", (idx, False)
        )

    def ensure_cols(self, sample_dict):
        self.data_schema = set(sample_dict.keys())

        cursor = self.con.cursor()
        cursor.execute(f"PRAGMA table_info({DATA_TABLE})")
        existing_columns = {row[1] for row in cursor.fetchall()}  # set of column names
        missing = set(sample_dict.keys()) - existing_columns

        for column in missing:
            value = sample_dict[column]
            col_type = infer_sqlite_type(value)
            assert col_type is not None
            sql = f"ALTER TABLE {DATA_TABLE} ADD COLUMN {column} {col_type}"
            cursor.execute(sql)

        self.con.commit()

    def add_success(self, idx, dicts):
        try:
            if len(dicts) > 0:
                new_dicts = []
                for d in dicts:
                    assert ID_FIELD not in d.keys()
                    new_d = {}
                    new_d[ID_FIELD] = idx
                    for k, v in d.items():
                        if infer_sqlite_type(v) is None:
                            new_d["__pickled_" + k] = pickle.dumps(v)
                        else:
                            new_d[k] = v
                    new_dicts.append(new_d)
                dicts = new_dicts

                if not self.cols_added:
                    self.ensure_cols(dicts[0])
                    self.cols_added = True

                for d in dicts:
                    assert self.data_schema == d.keys()

                columns = dicts[0].keys()
                column_names = ", ".join(columns)
                placeholders = ", ".join(f":{col}" for col in columns)
                sql = f"INSERT INTO {DATA_TABLE} ({column_names}) VALUES ({placeholders})"
                self.cur.executemany(sql, dicts)
        except (sqlite3.DataError, OverflowError) as e:
            logger.error(f"During processing of idx {idx} encountered")
            logger.error(e)
            self.add_failure(idx)
        else:
            self.cur.execute(
                f"INSERT INTO {PROCESSED_TABLE} ({ID_FIELD}, {SUCCESS_FIELD}) VALUES(?, ?)",
                (idx, True),
            )

    def process(self, ds, process_fn, process_fn_args):
        max_worklist_size = 300
        ray_wait_timeout = 1.0
        worklist = []

        while True:
            finished, worklist = ray.wait(worklist, timeout=ray_wait_timeout)

            while not self.should_break and len(worklist) < max_worklist_size:
                try:
                    idx, data = next(ds)
                except StopIteration:
                    break
                self.i += 1
                if idx not in self.already_processed:
                    self.already_processed.add(idx)
                    worklist.append(process_fn.remote(process_fn_args, idx, data))
                else:
                    logger.debug(f"Skipped {idx} because it was already processed.")

            for res in ray.get(finished):
                if res.data is None:
                    self.add_failure(res.i)
                else:
                    self.add_success(res.i, res.data)
            self.con.commit()

            if self.should_break_immediately:
                for task in worklist:
                    ray.cancel(task)
                break

            if len(worklist) == 0:
                break


def parse_args_and_run():
    parser = argparse.ArgumentParser(description="Generating inputs for ComPileLoop")

    parser.add_argument("--output-dataset", required=True)
    parser.add_argument("--output-dataset-json", required=True)

    parser.add_argument("--debug", default=False, action="store_true")

    args = parser.parse_args()
    main(args)


def main(args):
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    dw = DatasetWriter(args.output_dataset, args.output_dataset_json)
    max_idx = max(dw.already_processed)
    failed = set(range(max_idx)).difference(dw.already_processed)
    print(f"Processed idxs: {dw.already_processed}")
    print(f"Failed idxs: {failed}")
    print(f"Processed idxs len: {len(dw.already_processed)}")
    print(f"Failed idxs len: {len(failed)}")


if __name__ == "__main__":
    parse_args_and_run()
