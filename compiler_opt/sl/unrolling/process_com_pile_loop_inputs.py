#!/usr/bin/env python3

import argparse
import tempfile
import os
import json
import logging
import subprocess
import dataclasses
from datasets import load_dataset

from input_gen.input_gen import InputGenReplay, Input

import unrolling_runner

logger = logging.getLogger(__name__)

def parse_args_and_run():
    parser = argparse.ArgumentParser(
        description='Reading ComPileLoop'
    )
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--begin', default=0, type=int)
    parser.add_argument('--dump-llvm', default=False, action='store_true')

    parser.add_argument('--temp-dir', default=None)
    parser.add_argument('--save-temps', action='store_true', default=False)
    parser.add_argument('-mclang', default=[], action='append')
    parser.add_argument('-mllvm', default=[], action='append')

    parser.add_argument('-debug', default=False, action='store_true')

    args = parser.parse_args()
    main(args)

def main(args):
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    ds = load_dataset(args.dataset, split='train', streaming=True)
    ds = ds.skip(args.begin)
    for i, data in enumerate(ds):
        logger.info(f'Processing module {i}')
        process_module(data, args.dump_llvm, args)

def process_module(data, dump_llvm, args):

    inputs = [Input(**inpt) for inpt in data['inputs']]

    def to_dict(**kwargs):
        return kwargs

    replay_options = to_dict(
        working_dir=None,
        save_temps=args.save_temps,
        mclang=args.mclang,
        mllvm=args.mllvm,
        temp_dir=args.temp_dir,
    )

    process_and_args = [
        'opt', '-O3',
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        ud_samples = unrolling_runner.process_module(data['module'], process_and_args, tmpdir, inputs, replay_options, debug=args.debug)
        for uds in ud_samples:
            logging.info(f'Obtained sample {uds}')

if __name__ == '__main__':
    parse_args_and_run()
