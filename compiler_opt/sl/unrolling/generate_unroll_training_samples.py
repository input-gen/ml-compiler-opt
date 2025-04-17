#!/usr/bin/env python3

import pandas
import argparse
import tempfile
import os
import json
import logging
import subprocess
import ray
import dataclasses
from datasets import load_dataset

from input_gen.utils import InputGenReplay, Input, InputGenError
from com_pile_utils.dataset_writer import DatasetWriter, ProcessResult
from com_pile_utils.dataset_reader import DatasetReader

import unrolling_runner

logger = logging.getLogger(__name__)

def parse_args_and_run():
    parser = argparse.ArgumentParser(
        description='Reading ComPileLoop'
    )
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--output-dataset', required=True)

    parser.add_argument('--dump-llvm', default=False, action='store_true')

    parser.add_argument('--temp-dir', default=None)
    parser.add_argument('--save-temps', action='store_true', default=False)
    parser.add_argument('-mclang', default=[], action='append')
    parser.add_argument('-mllvm', default=[], action='append')

    parser.add_argument('--debug', default=False, action='store_true')

    args = parser.parse_args()
    main(args)

# 10MB
PARQUET_SIZE = 10 * 1000 * 1000

def main(args):
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    dr = DatasetReader(args.dataset)
    dw = DatasetWriter(args.output_dataset)
    dw.process(dr.get_iter(), process_module_wrapper, args)

@ray.remote
def process_module_wrapper(args, i, data):
    res = process_module(data, args.dump_llvm, args)
    if res is None:
        return ProcessResult(i, None)
    else:
        return ProcessResult(i, res)

def process_module(data, dump_llvm, args):

    inputs = data['inputs']
    if len(inputs) == 0:
        logger.debug('No inputs')
        return None
    if all([r is None for r in data['replays']]):
        logger.debug('No valid replays')
        return None

    def to_dict(**kwargs):
        return kwargs

    COMPILE_TIMEOUT = 1
    replay_options = to_dict(
        working_dir=None,
        save_temps=args.save_temps,
        mclang=args.mclang,
        mllvm=args.mllvm,
        temp_dir=args.temp_dir,
        compile_timeout=COMPILE_TIMEOUT,
    )

    process_and_args = [
        'opt', '-O3',
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            uch = unrolling_runner.UnrollCompilerHost(False, args.debug)

            decision_results = uch.get_unroll_decision_results(data['module'], process_and_args, tmpdir)

            samples = list(unrolling_runner.generate_samples(decision_results, inputs, replay_options))
            if len(samples) == 0:
                logger.debug('No samples generated')
                return None

            features_spec = uch.get_features_spec()
            advice_spec = uch.get_advice_spec()

            flattened_samples = []
            for features, advice in samples:
                flattened_features = []
                for feature in features:
                    assert(len(feature) == 1)
                    flattened_features.append(feature[0])
                flattened_samples.append(flattened_features + advice)

            labels = []
            for s in features_spec:
                labels.append(s.name)
            labels += [advice_spec.name + str(i + 2) for i in range(advice_spec.shape[0])]

            df = pandas.DataFrame(flattened_samples, columns=labels)
            logger.debug('Intermediate df')
            logger.debug(df)
            return df

        except InputGenError as e:
            logger.debug(f'InputGenGenerate failed: {e}')
            return None

if __name__ == '__main__':
    parse_args_and_run()
