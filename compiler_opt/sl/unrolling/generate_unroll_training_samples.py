#!/usr/bin/env python3

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


@dataclasses.dataclass(frozen=True)
class UnrollDecisionRuntime:
    factor: int
    runtime: Optional[List[int]]


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


PHYSICAL_CORE_RESOURCE = "physical_core"


def main(args):
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    # TODO Currently this is set up to only work with on a single node.
    # physical_cores = psutil.cpu_count(logical=False)
    # os.sched_setaffinity(0, [CPUS[physical_cores - 1]])
    # context = ray.init(resources={PHYSICAL_CORE_RESOURCE: physical_cores - 1})
    context = ray.init()
    print(f"Dashboard at {context.dashboard_url}")

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


CPUS = get_physical_cores()
HOSTNAME = socket.gethostname()
print(f"On host {HOSTNAME} cpus {CPUS}")
os.sched_setaffinity(0, [CPUS[len(CPUS) - 1]])

assert set(CPUS.keys()) == set(range(len(CPUS)))


@ray.remote(num_cpus=0, resources={PHYSICAL_CORE_RESOURCE: 1})
def benchmark_module(args, i, data):
    decision_results = data["decsion_results"]
    inputs = data["inputs"]

    hostname = socket.gethostname()
    print(f"HOSTNAME {HOSTNAME} hostname {hostname} cpus {CPUS}")
    core_id = ray.get_runtime_context().worker.core_worker.resource_ids()[PHYSICAL_CORE_RESOURCE][0][0]
    assert core_id < len(CPUS)
    os.sched_setaffinity(0, [CPUS[core_id]])

    def to_dict(**kwargs):
        return kwargs

    COMPILE_TIMEOUT = 2
    replay_options = to_dict(
        working_dir=None,
        save_temps=args.save_temps,
        mclang=args.mclang,
        mllvm=args.mllvm,
        temp_dir=args.temp_dir,
        compile_timeout=COMPILE_TIMEOUT,
    )

    try:
        samples = list(
            unrolling_runner.generate_samples(decision_results, inputs, replay_options, raw=True)
        )
    except InputGenError as e:
        return ProcessResult(i, None)

    if len(samples) == 0:
        logger.debug("No samples generated")
        return ProcessResult(i, None)
    if samples is None:
        logger.debug(f"InputGenReplay failed: {e}")
        return ProcessResult(i, None)

    d = {
        "features_spec": data["features_spec"],
        "advice_spec": data["advice_spec"],
        "samples": samples,
    }
    return ProcessResult(i, [d])


def get_speedup_factor(base: np.array, opt: np.array):
    # This will get element wise speedup factors for all inputs where either
    # succeeded
    arr = base / opt
    arr = arr[~np.isnan(arr)]  # remove NaNs
    if arr.size == 0:
        return None
    geomean = np.exp(np.mean(np.log(arr)))
    return geomean
    # return gmean(arr)


def rt_reduce(l):
    arr = np.array(l, dtype=float)
    arr = arr[~np.isnan(arr)]  # remove NaNs
    if arr.size == 0:
        return None
    return np.median(arr)
    # s = pd.Series(l).dropna()
    # if len(s) == 0:
    #     return None
    # return s.median()


def flatten(l):
    return np.fromiter((rt_reduce(sl) for sl in l), dtype=float)


def get_ud_sample_from_raw(x, base_runtime, factor_runtimes):
    with np.errstate(divide="ignore", invalid="ignore"):
        # Obtain speedup factors for all unroll factors.
        # Encode failure to unroll as speedup of 0.0.
        base_runtime = flatten(base_runtime)
        y = [
            get_speedup_factor(base_runtime, flatten(factor_runtime))
            if factor_runtime is not None
            else 0.0
            for factor_runtime in factor_runtimes
        ]

        # If we did not manage to obtain a speedup we fail
        if any(r is None for r in y):
            return None

        return (x, y)


def generate_samples(decision_results, inputs, replay_options, raw=False):
    def get_module_runtimes(module):
        with InputGenReplay(module, **replay_options) as igr:
            for inpt in inputs:
                num = 5
                timeout = 1
                rts = []
                for res in igr.replay_input(inpt.data, inpt.entry_no, num, timeout=timeout):
                    logger.debug(f"Res {res}")
                    re_match = re.search("MLGO_LOOP_UNROLL_TIMER ([0-9]+)", res.outs.decode("utf-8"))
                    if re_match is None:
                        logger.debug(f"No match")
                        rts.append(None)
                    else:
                        f = int(re_match.group(1))
                        logger.debug(f"Match {f}")
                        rts.append(f)
                yield rts

    def get_udr_runtime(udr: UnrollDecisionResult):
        if udr.action or udr.factor == 1:
            return UnrollDecisionRuntime(udr.factor, list(get_module_runtimes(udr.module)))
        else:
            return UnrollDecisionRuntime(udr.factor, None)

    def get_ud_sample(ud: UnrollDecision):
        res = get_ud_raw_sample(ud)
        if res is None:
            return None
        x, base_runtime, factor_runtimes = res
        return get_ud_sample_from_raw(x, base_runtime, factor_runtimes)

    def get_ud_raw_sample(ud: UnrollDecision):
        x = ud.features
        factor_runtimes = [None for _ in range(ADVICE_TENSOR_LEN)]
        for udr in ud.results:
            if udr.factor != 1:
                udrt = get_udr_runtime(udr)
                assert udrt.factor >= 2
                factor_runtimes[udrt.factor - UNROLL_FACTOR_OFFSET] = udrt.runtime

        logging.debug(f"Got factor_runtimes {factor_runtimes}")

        # If none of the factors succeeded.
        if all(factor_runtime is None for factor_runtime in factor_runtimes):
            return None

        # If we have any factor runtime to compare to, also get the base runtime
        base_runtime = None
        for udr in ud.results:
            if udr.factor == 1:
                udrt = get_udr_runtime(udr)
                base_runtime = udrt.runtime
                if base_runtime == None:
                    return None

        logging.debug(f"Got base_runtime {base_runtime}")

        return x, base_runtime, factor_runtimes

    def get_ud_raw_samples(uds: Iterable[UnrollDecision]):
        for ud in uds:
            sample = get_ud_raw_sample(ud)
            if sample is not None:
                yield sample
            else:
                logger.debug(f"Obtained invalid sample")

    def get_ud_samples(uds: Iterable[UnrollDecision]):
        for ud in uds:
            sample = get_ud_sample(ud)
            if sample is not None:
                yield sample
            else:
                logger.debug(f"Obtained invalid sample")

    if raw:
        yield from get_ud_raw_samples(decision_results)
    else:
        yield from get_ud_samples(decision_results)


if __name__ == "__main__":
    parse_args_and_run()
