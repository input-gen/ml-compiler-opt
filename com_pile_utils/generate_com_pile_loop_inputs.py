#!/usr/bin/env python3
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Tool for generating ComPileLoop+Inputs from ComPileLoop"""

import argparse
import tempfile
import re
import os
import json
import subprocess
import dataclasses
import collections
import sys
import stat
import logging
import glob
import pandas
import ray

from datasets import load_dataset
from typing import Dict, Tuple, BinaryIO, Union, List, Optional, Iterable

from input_gen.utils import (
    InputGenReplay,
    InputGenGenerate,
    Input,
    InputGenError,
    InputGenInstrumentationError,
)
from dataset_writer import DatasetWriter, ProcessResult

logger = logging.getLogger(__name__)

def parse_args_and_run():
    parser = argparse.ArgumentParser(description="Generating inputs for ComPileLoop")

    parser.add_argument("-mclang", default=[], action="append")
    parser.add_argument("-mllvm", default=[], action="append")

    parser.add_argument("--save-temps", action="store_true", default=False)
    parser.add_argument("--temp-dir", default=None)

    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dataset", required=True)
    parser.add_argument("--output-dataset-json", default=None)
    parser.add_argument("--begin", default=0, type=int)
    parser.add_argument("--end", default=None, type=int)

    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--debug-instrumentation", default=False, action="store_true")

    args = parser.parse_args()
    main(args)


def main(args):
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    # ray.init(log_to_driver=False)

    ds = load_dataset(args.dataset, split="train", streaming=True)
    dw = DatasetWriter(args.output_dataset, args.output_dataset_json, args.begin, args.end)
    dw.process(ds, process_module, args)


@ray.remote
def process_module_wrapper(args, i, data):
    return process_module(args, i, data)


def process_module(args, i, data):
    instrumentationLogger = logging.getLogger("input_gen_instrumentation_logger")
    if args.debug_instrumentation:
        instrumentationLogger.setLevel(logging.DEBUG)
    else:
        instrumentationLogger.setLevel(logging.WARNING)

    try:
        INPUTGEN_TIMEOUT = 1
        INPUTS_TO_GENERATE = 5

        def to_dict(**kwargs):
            return kwargs

        common_args = to_dict(
            working_dir=None,
            save_temps=args.save_temps,
            mclang=args.mclang,
            mllvm=args.mllvm,
            temp_dir=args.temp_dir,
        )

        igg = InputGenGenerate(
            data["module"],
            entries=["__llvm_extracted_loop"],
            **common_args,
        )
        assert igg.get_num_entries() == 1

        inputs = [
            dataclasses.asdict(i)
            for i in igg.generate(entry_no=0, num_inputs=INPUTS_TO_GENERATE, timeout=INPUTGEN_TIMEOUT)
        ]

        data["module"] = igg.get_repl_mod()

        logger.debug(data["module"])

        # TODO we want to gather some info on the inputs such as size, est. runtime,
        # We should also probably run the generated inputs and make sure they
        # run successfully.
        igr = InputGenReplay(
            data["module"],
            **common_args,
        )

        logger.debug(inputs)
        data["inputs_generated"] = len(inputs)

        inputs_normal_exit = []
        inputs_abnormal_exit = []
        for inpt in inputs:
            res = next(igr.replay_input(inpt["data"], entry_no=0, num=1, timeout=INPUTGEN_TIMEOUT))
            if res is None:
                inputs_abnormal_exit.append(inpt)
            else:
                inputs_normal_exit.append(inpt)

        size = len(data["module"])

        data["inputs_normal_exit"] = None
        data["inputs_abnormal_exit"] = None
        data["inputs_normal_exit_generated"] = len(inputs_normal_exit)
        data["inputs_abnormal_exit_generated"] = len(inputs_abnormal_exit)
        df = pandas.DataFrame(data, index=[0])
        df.at[0, "inputs_normal_exit"] = inputs_normal_exit
        df.at[0, "inputs_abnormal_exit"] = inputs_abnormal_exit

        return ProcessResult(df, size, i)

    except InputGenInstrumentationError as e:
        instrumentationLogger.debug(f"Instrumentation error in module {i}")
        instrumentationLogger.debug(e)
        return None
    except InputGenError as e:
        logger.debug(f"InputGenGenerate failed: {e}")
        return None


if __name__ == "__main__":
    parse_args_and_run()
