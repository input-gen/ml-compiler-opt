#!/usr/bin/env python3
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Tool for generating ComPileLoop from ComPile"""

import argparse
import os
import shutil
import tempfile
import subprocess
import json
import ray
import logging

from .dataset_writer import DatasetWriter, ProcessResult
from .dataset_reader import DatasetReader

logger = logging.getLogger(__name__)


def parse_args_and_run():
    parser = argparse.ArgumentParser(description="A tool for making a LLVM IR loop dataset")

    # parser.add_argument("--language", default="c")

    parser.add_argument("--save-temps", action="store_true", default=False)
    parser.add_argument("--temp-dir", default=None)

    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dataset", required=True)

    parser.add_argument("--one", type=int, default=None)

    parser.add_argument("--debug", default=False, action="store_true")

    args = parser.parse_args()

    # args.dataset = os.path.join(args.dataset, args.language)

    main(args)


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
def process_module_wrapper(args, i, data):
    return process_module_impl(data["content"], data["language"], i, args)


def process_module(args, i, data):
    return process_module_impl(data["content"], data["language"], i, args)


def process_module_impl(module, language, idx, args):
    try:
        outdir = tempfile.mkdtemp(dir=args.temp_dir)
        if args.save_temps:
            with open(os.path.join(outdir, "module.bc"), "wb") as f:
                f.write(module)
        return process_module_in_dir(module, language, idx, outdir)
    finally:
        if not args.save_temps:
            shutil.rmtree(outdir)


def process_module_in_dir(module, language, idx, temp_outdir):
    metadata_filename = str(os.path.join(temp_outdir, "metadata.json"))
    cmd = [
        "llvm-extract-loops",
        "-",
        "-o",
        "-",
        "--metadata",
        metadata_filename,
    ]
    logger.debug(" ".join(cmd))
    with subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE
    ) as proc:
        outs, errs = proc.communicate(input=module)
        if proc.wait() != 0:
            logger.debug("llvm-extract-loops failed")
            logger.debug("outs")
            logger.debug(outs.decode("utf-8"))
            logger.debug("errs")
            logger.debug(errs.decode("utf-8"))
            return ProcessResult(idx, None)

    with open(metadata_filename, "r") as metadata_file:
        data = json.load(metadata_file)

    if data["num_loops"] == 0:
        return ProcessResult(idx, [])

    data["module"] = outs
    data["language_in_compile"] = language

    return ProcessResult(idx, [data])


if __name__ == "__main__":
    parse_args_and_run()
