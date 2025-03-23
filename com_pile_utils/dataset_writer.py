#!/usr/bin/env python3

import ray
import argparse
import json
import signal
import pandas
import pyarrow
import os
import dataclasses

from pyarrow import parquet
from datasets import load_dataset

# 100MB
PARQUET_SIZE = 100 * 1000 * 1000
VER = '1.1'

@dataclasses.dataclass(frozen=True)
class ProcessResult:
    df: pandas.DataFrame
    size: int
    i: int

class DatasetWriter:
    def __init__(self, begin, end, parquet_start, output_dataset, output_dataset_json, parquet_size=PARQUET_SIZE):
        self.output_dataset = output_dataset
        self.output_dataset_json = output_dataset_json
        self.parquet_size = parquet_size

        self.begin = begin
        self.end = end

        self.dfs = []
        self.idxs_in_parquet = []
        self.total_pfile_size = 0
        self.parquet_id = parquet_start
        self.should_break = False
        self.i = begin

        signal.signal(signal.SIGUSR2, self.receive_should_break)
        signal.signal(signal.SIGUSR1, self.receive)

    def receive(self, signum, stack):
        print(f'Progress: module {self.i} size {self.total_pfile_size} pending {sorted(self.idxs_in_parquet)}')

    def receive_should_break(self, signum, stack):
        print(f'Will break')
        self.should_break = True

    def get_current_parquet_name(self):
        return os.path.join(self.output_dataset, 'train-' + str(self.parquet_id) + '.parquet')

    def get_current_parquet_json_name(self):
        return os.path.join(self.output_dataset_json, 'train-' + str(self.parquet_id) + '.parquet.json')

    def write_parquet(self):
        name = self.get_current_parquet_name()
        json_name = self.get_current_parquet_json_name()
        if len(self.dfs) == 0:
            return
        print(f'Writing intermediate parquet {self.parquet_id} with estimated size {self.total_pfile_size} for modules {sorted(self.idxs_in_parquet)}')
        df = pandas.concat(self.dfs)
        table = pyarrow.Table.from_pandas(df, preserve_index=False)
        parquet.write_table(table, name, compression='NONE')
        with open(json_name, 'w') as fp:
            fp.write(json.dumps({
                'version' : VER,
                'estimated_size' : self.total_pfile_size,
                'num' : len(self.idxs_in_parquet),
                'idxs' : sorted(self.idxs_in_parquet),
            }, indent=4) + '\n')

        self.dfs = []
        self.idxs_in_parquet = []
        self.total_pfile_size = 0
        self.parquet_id += 1

    def process(self, ds, process_fn, process_fn_args):
        curparname = self.get_current_parquet_name()
        if os.path.exists(curparname):
            raise Exception(f'The parquet file {curparname} already exists. Aborting.')
        curjsonname = self.get_current_parquet_json_name()
        if os.path.exists(curjsonname):
            raise Exception(f'The parquet json file {curjsonname} already exists. Aborting.')

        os.makedirs(self.output_dataset, exist_ok=True)
        os.makedirs(self.output_dataset_json, exist_ok=True)

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
                worklist.append(process_fn.remote(process_fn_args, self.i, data))
                self.i += 1

            for res in ray.get(finished):
                if res is None:
                    continue
                self.total_pfile_size += res.size
                self.dfs.append(res.df)
                self.idxs_in_parquet.append(res.i)
                if self.total_pfile_size > PARQUET_SIZE:
                    self.write_parquet()

            if len(worklist) == 0:
                break

        print(f'Writing final parquet {self.i}')
        self.write_parquet()

if __name__ == '__main__':
    parse_args_and_run()
