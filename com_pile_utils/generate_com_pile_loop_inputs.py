#!/usr/bin/env python3
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Tool for generating ComPileLoop+Inputs from ComPileLoop"""

import random
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

    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--debug-instrumentation", default=False, action="store_true")

    args = parser.parse_args()
    main(args)


def iter_dataset(ds):
    i = 0
    for d in ds:
        yield (i, d)
        i += 1


def main(args):
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    # ray.init(log_to_driver=False)

    ds = load_dataset(args.dataset, split="train", streaming=True)
    dw = DatasetWriter(args.output_dataset)
    dw.process(iter_dataset(ds), process_module_wrapper, args)


@ray.remote
def process_module_wrapper(args, idx, data):
    return process_module(args, idx, data)


def process_module(args, idx, data):
    instrumentationLogger = logging.getLogger("input_gen_instrumentation_logger")
    if args.debug_instrumentation:
        instrumentationLogger.setLevel(logging.DEBUG)
    else:
        instrumentationLogger.setLevel(logging.WARNING)

    try:
        COMPILE_TIMEOUT = 1
        INPUTGEN_TIMEOUT = 1
        NUM_INPUTS = 5
        # tuples of (int_min, int_max, num_inputs)
        INPUTGEN_STRATEGY = [
            (20, 40, NUM_INPUTS),
            (-40, 40, NUM_INPUTS),
            (64, 128, NUM_INPUTS),
            (-100, 128, NUM_INPUTS),
            (500, 1000, NUM_INPUTS),
            (-1000, 1000, NUM_INPUTS),
        ]

        INPUTS_TO_GENERATE = 5

        def to_dict(**kwargs):
            return kwargs

        common_args = to_dict(
            working_dir=None,
            save_temps=args.save_temps,
            mclang=args.mclang,
            mllvm=args.mllvm,
            temp_dir=args.temp_dir,
            compile_timeout=COMPILE_TIMEOUT,
        )

        igg = InputGenGenerate(
            data["module"],
            entries=["__llvm_extracted_loop"],
            **common_args,
        )
        assert igg.get_num_entries() == 1

        inputs = []
        for int_min, int_max, num_inputs in INPUTGEN_STRATEGY:
            # We do a separate igg.generate for each single input because we
            # want different seeds for each one.
            for i in range(num_inputs):
                # 0 to int32_t_max
                seed = random.randint(0, 2147483647)
                try:
                    inputs += [
                        dataclasses.asdict(i)
                        for i in igg.generate(
                            entry_no=0,
                            num_inputs=1,
                            first_input=i,
                            timeout=INPUTGEN_TIMEOUT,
                            int_min=int_min,
                            int_max=int_max,
                            seed=seed,
                        )
                    ]
                except InputGenError as e:
                    logger.debug(e)

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
        data["inputs_generated_num"] = len(inputs)

        inputs_normal_exit = []
        inputs_abnormal_exit = []
        for inpt in inputs:
            res = next(igr.replay_input(inpt["data"], entry_no=0, num=1, timeout=INPUTGEN_TIMEOUT))
            if res is None or "replay" not in res.timers:
                inputs_abnormal_exit.append(inpt)
            else:
                inpt["replay_time"] = res.timers["replay"]
                inputs_normal_exit.append(inpt)

        data["inputs_normal_exit"] = inputs_normal_exit
        data["inputs_abnormal_exit"] = inputs_abnormal_exit
        data["inputs_normal_exit_generated_num"] = len(inputs_normal_exit)
        data["inputs_abnormal_exit_generated_num"] = len(inputs_abnormal_exit)

        return ProcessResult(idx, [data])

    except InputGenInstrumentationError as e:
        instrumentationLogger.debug(f"Instrumentation error in module {idx}")
        instrumentationLogger.debug(e)
        return ProcessResult(idx, None)
    except InputGenError as e:
        logger.debug(f"InputGenGenerate failed in module {idx}")
        logger.debug(e)
        return ProcessResult(idx, None)


if __name__ == "__main__":
    parse_args_and_run()
