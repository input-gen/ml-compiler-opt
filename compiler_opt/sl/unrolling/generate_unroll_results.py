import pandas
import psutil
import argparse
import tempfile
import os
import json
import logging
import subprocess
import ray
import dataclasses
from datasets import load_dataset
import socket


from input_gen.utils import Input, InputGenError
from com_pile_utils.dataset_writer import DatasetWriter, ProcessResult
from com_pile_utils.dataset_reader import DatasetReader

from . import unrolling_runner

logger = logging.getLogger(__name__)


def parse_args_and_run():
    parser = argparse.ArgumentParser(description="Reading ComPileLoop")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dataset", required=True)

    parser.add_argument("--dump-llvm", default=False, action="store_true")

    parser.add_argument("--temp-dir", default=None)
    parser.add_argument("--save-temps", action="store_true", default=False)
    parser.add_argument("-mclang", default=[], action="append")
    parser.add_argument("-mllvm", default=[], action="append")

    parser.add_argument("--one", type=int, default=None)

    parser.add_argument("--debug", default=False, action="store_true")

    args = parser.parse_args()
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
            data = next(it)[1]
            process_module(args, args.one, data)


def get_physical_cores():
    output = subprocess.check_output("lscpu -p=CPU,Core", shell=True).decode()
    lines = [line for line in output.splitlines() if not line.startswith("#")]
    mapping = {}
    for line in lines:
        cpu, core = map(int, line.split(","))
        if core not in mapping:
            mapping[core] = cpu
    return mapping


@ray.remote
def process_module_wrapper(args, i, data):
    res = process_module(args, i, data)
    if res is None:
        return ProcessResult(i, None)
    else:
        return ProcessResult(i, res)


def process_module(args, idx, data):
    if args.save_temps:
        with tempfile.NamedTemporaryFile(mode="wb", delete=False) as tmp:
            with open(tmp.name, "wb") as f:
                f.write(data["module"])
            logger.debug(f"Wrote tmp module to {tmp.name}")
    dump_llvm = args.dump_llvm
    inputs = data["inputs"]
    if len(inputs) == 0:
        logger.debug("No inputs")
        return None
    # Let's allow invalid? replays just in case.
    if False and all([r is None for r in data["replays"]]):
        logger.debug("No valid replays")
        return None

    process_and_args = [
        "opt",
        "-O3",
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        uch = unrolling_runner.UnrollCompilerHost(False, args.debug)

        decision_results = list(
            uch.get_unroll_decision_results(data["module"], process_and_args, tmpdir)
        )

    if len(decision_results) == 0:
        return None

    features_spec = uch.get_features_spec()
    advice_spec = uch.get_advice_spec()

    d = {
        "features_spec": features_spec,
        "advice_spec": advice_spec,
        "decision_results": decision_results,
        "inputs": inputs,
        "replays": data["replays"],
        "num_loops": data["num_loops"],
    }

    logger.debug("Got unroll result")
    logger.debug(d)

    return [d]


if __name__ == "__main__":
    parse_args_and_run()
