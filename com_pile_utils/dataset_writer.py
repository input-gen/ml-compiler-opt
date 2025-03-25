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

from pyarrow import parquet
from datasets import load_dataset

logger = logging.getLogger(__name__)

# 100MB
PARQUET_SIZE = 100 * 1000 * 1000
VER = "1"


@dataclasses.dataclass(frozen=True)
class ProcessResult:
    df: pandas.DataFrame
    size: int
    i: int


class VersionMismatch(Exception):
    expected: str
    found: str
    pass


class CorruptMetadata(Exception):
    pass


class DatasetWriter:
    def __init__(
        self, output_dataset, output_dataset_json=None, begin=0, end=None, parquet_size=PARQUET_SIZE
    ):
        if output_dataset_json is None:
            split = os.path.split(output_dataset)
            if split[1] == "":
                base = split[0]
            else:
                base = output_dataset
            output_dataset_json = base + "Json"

        self.output_dataset = output_dataset
        self.output_dataset_json = output_dataset_json
        self.parquet_size = parquet_size

        self.begin = begin
        self.end = end

        self.dfs = []
        self.idxs_in_parquet = []
        self.total_pfile_size = 0
        self.parquet_id = None
        self.should_break = False
        self.should_break_immediately = False
        self.i = begin

        self.parse_metadata()

        signal.signal(signal.SIGUSR2, self.receive)
        signal.signal(signal.SIGUSR1, self.receive)
        signal.signal(signal.SIGINT, self.receive)

    def parse_metadata(self):
        os.makedirs(self.output_dataset, exist_ok=True)
        os.makedirs(self.output_dataset_json, exist_ok=True)

        self.already_processed = []

        json_filenames = glob.glob(os.path.join(self.output_dataset_json, "train-*.parquet.json"))
        json_re = re.compile(r"train-([0-9]+).parquet.json")
        parquet_re = re.compile(r"train-([0-9]+).parquet")

        json_idxs = []
        for bn in os.listdir(self.output_dataset_json):
            name_match = json_re.match(bn)
            if name_match is None:
                logger.debug(f"Found non matching json name {bn}")
                continue
            json_idx = int(name_match.group(1))
            logger.debug(f"Found matching json name {bn} with idx {json_idx}")
            json_idxs.append(json_idx)

            json_filename = os.path.join(self.output_dataset_json, bn)
            parquet_filename = os.path.join(self.output_dataset, bn[:-5])
            logger.debug(f"Checking for parquet {parquet_filename}")
            if not os.path.exists(parquet_filename):
                raise CorruptMetadata("Corresponding parquet does not exist")
            with open(json_filename, "r") as f:
                try:
                    obj = json.load(f)
                except JSONDecodeError as e:
                    raise CorruptMetadata(str(e))
                if obj["version"] != VER:
                    raise VersionMismatch(VER, obj["version"])
                logger.debug(f"Read metadate {obj}")
                if len(obj["idxs"]) != obj["num"]:
                    raise CorruptMetadata("Length mismatch")
                self.already_processed += obj["idxs"]

        json_idxs.sort()
        if json_idxs != list(range(len(json_idxs))):
            raise CorruptMetadata(f"Missing jsons {json_idx}")

        parquet_idxs = []
        for bn in os.listdir(self.output_dataset_json):
            name_match = parquet_re.match(bn)
            if name_match is None:
                logger.debug(f"Found non matching parquet name {bn}")
                continue
            parquet_idx = int(name_match.group(1))
            logger.debug(f"Found matching parquet name {bn} with idx {parquet_idx}")
            parquet_idxs.append(parquet_idx)

        parquet_idxs.sort()
        if json_idxs != parquet_idxs:
            raise CorruptMetadata(f"Mismatch between jsons and parquets {json_idxs}, {parquet_idxs}")

        self.already_processed = set(self.already_processed)
        self.parquet_id = len(parquet_idxs)

        if len(self.already_processed) == 0:
            logger.info("Starting a fresh dataset.")
        else:
            logger.info(f"Found an in-progress dataset. Will continue from parquet {self.parquet_id}.")

    def receive(self, signum, stack):
        if signum == signal.SIGUSR1:
            print(
                f"Progress: module {self.i} size {self.total_pfile_size} pending {sorted(self.idxs_in_parquet)}"
            )
        elif signum == signal.SIGUSR2:
            print("Will break")
            self.should_break = True
        elif signum == signal.SIGINT:
            print("Will break immediately")
            self.should_break = True
            self.should_break_immediately = True

    def get_current_parquet_name(self):
        return os.path.join(self.output_dataset, "train-" + str(self.parquet_id) + ".parquet")

    def get_current_parquet_json_name(self):
        return os.path.join(
            self.output_dataset_json, "train-" + str(self.parquet_id) + ".parquet.json"
        )

    def write_parquet(self):
        name = self.get_current_parquet_name()
        json_name = self.get_current_parquet_json_name()
        assert not os.path.exists(name)
        assert not os.path.exists(json_name)
        if len(self.dfs) == 0:
            return
        print(
            f"Writing parquet {self.parquet_id} with estimated size {self.total_pfile_size} for modules {sorted(self.idxs_in_parquet)}"
        )
        df = pandas.concat(self.dfs)
        table = pyarrow.Table.from_pandas(df, preserve_index=False)
        parquet.write_table(table, name, compression="NONE")
        with open(json_name, "w") as fp:
            fp.write(
                json.dumps(
                    {
                        "version": VER,
                        "estimated_size": self.total_pfile_size,
                        "num": len(self.idxs_in_parquet),
                        "idxs": sorted(self.idxs_in_parquet),
                    },
                    indent=4,
                )
                + "\n"
            )

        self.dfs = []
        self.idxs_in_parquet = []
        self.total_pfile_size = 0
        self.parquet_id += 1

    def process(self, ds, process_fn, process_fn_args):
        ds = iter(ds.skip(self.i))

        max_worklist_size = 300
        ray_wait_timeout = 1.0
        worklist = []

        while True:
            finished, worklist = ray.wait(worklist, timeout=ray_wait_timeout)

            while not self.should_break and len(worklist) < max_worklist_size:
                if self.i == self.end:
                    break
                try:
                    data = next(ds)
                except StopIteration:
                    break
                if self.i not in self.already_processed:
                    worklist.append(process_fn.remote(process_fn_args, self.i, data))
                else:
                    logger.debug(f"Skipped {self.i} because it was already processed.")
                self.i += 1

            for res in ray.get(finished):
                if res is None:
                    continue
                self.total_pfile_size += res.size
                self.dfs.append(res.df)
                self.idxs_in_parquet.append(res.i)
                if self.total_pfile_size > self.parquet_size:
                    self.write_parquet()

            if self.should_break_immediately:
                for task in worklist:
                    ray.cancel(task)
                break

            if len(worklist) == 0:
                break

        self.write_parquet()


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
