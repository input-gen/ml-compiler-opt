#!/usr/bin/env python3
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Tool for generating ComPileLoop+Inputs from ComPileLoop"""

import random
import argparse
import logging
import ray

from typing import Dict, Tuple, BinaryIO, Union, List, Optional, Iterable

from input_gen.utils import (
    InputGenReplay,
    InputGenGenerate,
    Input,
    InputGenError,
    InputGenInstrumentationError,
)
from .dataset_writer import ProcessResult, ID_FIELD
from . import generate_main

logger = logging.getLogger(__name__)


def parse_args_and_run():
    parser = argparse.ArgumentParser(description="Generating inputs for ComPileLoop")

    parser.add_argument("-mclang", default=[], action="append")
    parser.add_argument("-mllvm", default=[], action="append")

    parser.add_argument("--save-temps", action="store_true", default=False)
    parser.add_argument("--temp-dir", default=None)

    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dataset", required=True)

    parser.add_argument("--one", type=int, default=None)

    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--debug-instrumentation", default=False, action="store_true")

    args = parser.parse_args()
    generate_main.main(args, process_module_wrapper, process_module)


def main(args):
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    with DatasetReader(args.dataset) as dr:
        if args.one is None:
            with DatasetWriter(args.output_dataset) as dw:
                dw.process(dr.get_iter(), process_module_wrapper, args)
        else:
            it = dr.get_one_iter(args.one)
            process_module(args, args.one, next(it)[1])


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
        COMPILE_TIMEOUT = 2
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

        num_loops = data["num_loops"]
        assert num_loops > 0

        entries = []
        for i in range(num_loops):
            entries.append("__llvm_extracted_loop." + str(i))

        with InputGenGenerate(
            data["module"],
            entries=entries,
            **common_args,
        ) as igg:
            assert igg.get_num_entries() == num_loops
            data["module"] = igg.get_repl_mod()

            inputs = []
            for i in range(num_loops):
                entry_inputs = []
                for int_min, int_max, num_inputs in INPUTGEN_STRATEGY:
                    # We do a separate igg.generate for each single input because we
                    # want different seeds for each one.
                    for j in range(num_inputs):
                        # 0 to int32_t_max
                        seed = random.randint(0, 2147483647)
                        try:
                            entry_inputs += igg.generate(
                                entry_no=i,
                                num_inputs=1,
                                first_input=j,
                                timeout=INPUTGEN_TIMEOUT,
                                int_min=int_min,
                                int_max=int_max,
                                seed=seed,
                            )
                        except InputGenError as e:
                            logger.debug(e)
                inputs.append(entry_inputs)
            data["inputs"] = inputs

        # TODO we want to gather some info on the inputs such as size, est. runtime,
        with InputGenReplay(
            data["module"],
            **common_args,
        ) as igr:
            replays = []
            for entry_inputs in inputs:
                entry_replays = []
                for inpt in entry_inputs:
                    res = next(
                        igr.replay_input(inpt.data, entry_no=0, num=1, timeout=INPUTGEN_TIMEOUT)
                    )
                    replays.append(res)
                replays.append(entry_replays)
        data["replays"] = replays

        logger.debug(data)

        del data[ID_FIELD]
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
