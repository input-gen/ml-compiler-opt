import pandas
import psutil
import re
import argparse
import tempfile
import os
import json
import logging
import subprocess
import ray
import dataclasses
import numpy as np
from scipy import stats
from typing import Dict, Tuple, BinaryIO, Union, List, Optional, Iterable


from input_gen.utils import Input, InputGenError, InputGenReplay
from com_pile_utils.dataset_writer import DatasetWriter, ProcessResult
from com_pile_utils.dataset_reader import DatasetReader

from . import unrolling_runner
from .unrolling_runner import UnrollFactorResult, UnrollDecision
from .unroll_model import ADVICE_TENSOR_LEN, UNROLL_FACTOR_OFFSET, MAX_UNROLL_FACTOR
from .datastructures import *

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
    parser.add_argument("--debug-profiling", default=False, action="store_true")

    args = parser.parse_args()
    main(args)


PHYSICAL_CORE_RESOURCE = "physical_core"

MANAGER_CORES_NUM = 5


def main(args):
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    with DatasetReader(args.dataset) as dr:
        if args.one is None:
            physical_cores = psutil.cpu_count(logical=False)
            cpu_mapping = get_physical_cores()
            assert len(cpu_mapping) == physical_cores
            logger.info(f"cpu_mapping {cpu_mapping}")

            manager_cores = list(range(physical_cores - 1 - MANAGER_CORES_NUM, physical_cores))
            logger.info(f"Manager cores {manager_cores}")

            manager_cpus = sum([cpu_mapping[i] for i in manager_cores], [])
            logger.info(f"Manager cpus {manager_cpus}")
            os.sched_setaffinity(0, manager_cpus)

            context = ray.init(
                log_to_driver=False, resources={PHYSICAL_CORE_RESOURCE: physical_cores - 1}
            )
            args.cpu_mapping = ray.get(get_physical_cpu_mapping.remote())
            assert physical_cores == len(args.cpu_mapping)
            assert args.cpu_mapping == cpu_mapping

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
            mapping[core] = [cpu]
        else:
            mapping[core].append(cpu)
    return mapping


@ray.remote(num_cpus=0, resources={PHYSICAL_CORE_RESOURCE: 1})
def get_physical_cpu_mapping():
    return get_physical_cores()


@ray.remote(num_cpus=0, resources={PHYSICAL_CORE_RESOURCE: 1})
def process_module_wrapper(args, i, data):
    logger = logging.getLogger(__name__ + ".process_module_wrapper")
    if args.debug or args.debug_profiling:
        logger.setLevel(level=logging.DEBUG)

    logger.debug(f"cpus {args.cpu_mapping}")
    core_id = ray.get_runtime_context().worker.core_worker.resource_ids()[PHYSICAL_CORE_RESOURCE][0][0]
    assert core_id < len(args.cpu_mapping) - 1
    benchmarking_cores = [args.cpu_mapping[core_id][0]]
    task_cores = args.cpu_mapping[core_id]
    logger.debug(f"benchmarking {benchmarking_cores}")
    logger.debug(f"task {task_cores}")

    os.sched_setaffinity(0, task_cores)
    res = process_module_impl_results(args, i, data)
    if res is None:
        return ProcessResult(i, None)
    assert len(res) == 1
    os.sched_setaffinity(0, benchmarking_cores)
    return process_module_impl_sample(args, i, res[0])


def process_module(args, i, data):
    res = process_module_impl_results(args, i, data)
    if res is None:
        return ProcessResult(i, None)
    else:
        assert len(res) == 1
        return process_module_impl_sample(args, i, res[0])


def process_module_impl_results(args, idx, data):
    logger = logging.getLogger(__name__ + ".process_module_impl_results")
    if args.debug or args.debug_profiling:
        logger.setLevel(level=logging.DEBUG)

    if args.save_temps:
        with tempfile.NamedTemporaryFile(mode="wb", delete=False) as tmp:
            with open(tmp.name, "wb") as f:
                f.write(data["module"])
            logger.debug(f"Wrote tmp module to {tmp.name}")
    dump_llvm = args.dump_llvm
    inputs = data["inputs"]
    num_inputs = sum(map(len, inputs))
    if num_inputs == 0:
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


def process_module_impl_sample(args, i, data):
    logger = logging.getLogger(__name__ + ".process_module_impl_sample")
    if args.debug or args.debug_profiling:
        logger.setLevel(level=logging.DEBUG)

    decision_results = data["decision_results"]
    inputs = data["inputs"]

    def to_dict(**kwargs):
        return kwargs

    COMPILE_TIMEOUT = 2
    clang_flags = ["./compiler_opt/sl/unrolling/rts/unrolling_profiler.o", "-lpfm"]
    replay_options = to_dict(
        working_dir=None,
        save_temps=args.save_temps,
        mclang=args.mclang + clang_flags,
        mllvm=args.mllvm,
        temp_dir=args.temp_dir,
        compile_timeout=COMPILE_TIMEOUT,
    )

    logger.debug(f"Attempting sample generation for {len(decision_results)} udr's")

    samples = list(generate_samples(decision_results, inputs, replay_options, args))

    if len(samples) == 0:
        logger.debug("No samples generated")
        return ProcessResult(i, None)

    logger.debug(f"Successfully generated {len(samples)} samples")

    d = {
        "features_spec": data["features_spec"],
        "advice_spec": data["advice_spec"],
        "samples": samples,
    }
    return ProcessResult(i, [d])


INITIAL_SAMPLES = 5
MAX_INITIAL_SAMPLES = 10
MAX_SAMPLES = 25
CONFIDENCE = 0.95
RELATIVE_CI_THRESHOLD = 0.05


def get_benchmarking_mean_ci(samples, confidence):
    if len(samples) == 0:
        return np.nan, np.inf
    if len(samples) == 1:
        return samples[0], np.inf

    sample_mean = np.mean(samples)
    sample_std = np.std(samples, ddof=1)  # sample std (unbiased)

    # t critical value
    n = len(samples)
    alpha = 1 - confidence
    t_crit = stats.t.ppf(1 - alpha / 2, df=n - 1)

    margin_error = t_crit * (sample_std / np.sqrt(n))
    relative_ci_width = (2 * margin_error) / sample_mean
    return sample_mean, relative_ci_width


def adaptive_benchmark(
    iterator,
    initial_samples=INITIAL_SAMPLES,
    max_initial_samples=MAX_INITIAL_SAMPLES,
    max_samples=MAX_SAMPLES,
    confidence=CONFIDENCE,
    relative_ci_threshold=RELATIVE_CI_THRESHOLD,
    logger=logger,
    fail_on_non_convergence=False,
):
    """
    Adaptive benchmarking loop to estimate mean runtime with confidence.

    Parameters:
        iterator: An iterator yielding runtime samples.
        initial_samples: Number of initial samples to take.
        max_initial_samples: Max number of tries to get initial samples.
        max_samples: Max number of samples to avoid infinite loop.
        confidence: Desired confidence level (e.g., 0.95 for 95% CI).
        relative_ci_threshold: Target relative CI width (e.g., 0.05 means CI width < 5% of mean).

    Returns:
        AdaptiveBenchmarkingResult
    """
    assert max_initial_samples < max_samples

    logger.debug("Starting adaptive benchmarking")

    samples = np.array([], dtype=float)

    n = 0
    while len(samples) < initial_samples and n < max_initial_samples:
        new_sample = next(iterator)
        if new_sample is not None:
            new_sample = float(new_sample)
            samples = np.append(samples, new_sample)
            if n == 0 and new_sample == 0:
                logger.debug(f"Got zero")
                return get_zero_rt_abr()
            logger.debug(f"Obtained sample {new_sample}, len {len(samples)}")
        n += 1

    if len(samples) < initial_samples:
        logger.debug(f"Too many replay failures")
        sample_mean, relative_ci_width = get_benchmarking_mean_ci(samples, confidence)
        return AdaptiveBenchmarkingResult(samples, sample_mean, relative_ci_width, False)

    assert n < max_samples

    while n < max_samples:
        sample_mean, relative_ci_width = get_benchmarking_mean_ci(samples, confidence)

        if relative_ci_width < relative_ci_threshold:
            logger.debug(f"Converged: mean {sample_mean}, ci {relative_ci_width}")
            return AdaptiveBenchmarkingResult(samples, sample_mean, relative_ci_width, True)

        new_sample = None
        while new_sample is None and n < max_samples:
            new_sample = next(iterator)
            n += 1
        if new_sample is not None:
            samples = np.append(samples, float(new_sample))

    logger.debug(f"Did not converge: mean {sample_mean}, ci {relative_ci_width}")

    if fail_on_non_convergence:
        return get_invalid_abr()
    else:
        return AdaptiveBenchmarkingResult(samples, sample_mean, relative_ci_width, False)


def invalidate_high_variance_rts(
    rts,
    cis,
    relative_ci_threshold=RELATIVE_CI_THRESHOLD,
):
    if rts is not None:
        assert cis is not None
        rts[cis > relative_ci_threshold] = np.nan


def get_speedup_factor(base: np.array, opt: np.array):
    # This will get element wise speedup factors for all inputs where either
    # succeeded
    arr = base / opt
    arr = arr[~np.isnan(arr)]  # remove NaNs
    if arr.size == 0:
        return None
    geomean = np.exp(np.mean(np.log(arr)))
    return geomean


def get_ud_sample_from_raw(
    udrs,
    relative_ci_threshold=RELATIVE_CI_THRESHOLD,
    logger=logger,
):
    x = udrs.features
    base_runtime = udrs.base_runtime
    base_ci = udrs.base_ci
    invalidate_high_variance_rts(base_runtime, base_ci, relative_ci_threshold)
    factor_runtimes = udrs.factor_runtimes
    factor_cis = udrs.factor_cis
    for rts, cis in zip(factor_runtimes, factor_cis):
        invalidate_high_variance_rts(rts, cis, relative_ci_threshold)
    with np.errstate(divide="ignore", invalid="ignore"):
        # Obtain speedup factors for all unroll factors.
        # Encode failure to unroll as speedup of 0.0.
        y = [
            get_speedup_factor(base_runtime, factor_runtime) if factor_runtime is not None else 0.0
            for factor_runtime in factor_runtimes
        ]

        # If we did not manage to obtain a speedup we fail
        if any(r is None for r in y):
            logger.debug(f"Failed to obtain speedup for {len([r is None for r in y])}")
            return None

        return UnrollDecisionTrainingSample(x, np.array(y))


def filter_none(l):
    return filter(lambda x: x is not None, l)


def get_maybe_non_zero_runtime_array(abrs: List[AdaptiveBenchmarkingResult]):
    non_zero_runtime = np.ones(len(abrs), dtype=bool)
    for i, abr in enumerate(abrs):
        if abr.is_zero_rt():
            non_zero_runtime[i] = 0
    return non_zero_runtime


def generate_samples(decision_results, inputs: List[List], replay_options, args, raw=True):
    logger = logging.getLogger(__name__ + ".generate_samples")
    if args.debug or args.debug_profiling:
        logger.setLevel(level=logging.DEBUG)

    num_inputs = sum(map(len, inputs))

    def get_rt_from_replay_res(res):
        logger.debug(f"Res {res}")
        re_match = re.search("MLGO_LOOP_UNROLL_TIMER ([0-9]+)", res.outs.decode("utf-8"))
        if re_match is None:
            logger.debug(f"No match")
            return None
        else:
            f = int(re_match.group(1))
            logger.debug(f"Match {f}")
            return f

    def get_module_runtimes(module, maybe_non_zero_runtime=None):
        NUM_REPLAYS = None
        TIMEOUT = 1

        rtss = []
        i = 0
        with InputGenReplay(module, **replay_options) as igr:
            logger.debug(f"Starting replaying {len(inputs)} entries")
            for entry_no, entry_inputs in enumerate(inputs):
                logger.debug(
                    f"Starting replaying for entry {entry_no} with {len(entry_inputs)} inputs"
                )
                for input_no, inpt in enumerate(entry_inputs):
                    if maybe_non_zero_runtime is None or maybe_non_zero_runtime[i]:
                        try:
                            it = igr.replay_input(
                                inpt.data, inpt.entry_no, NUM_REPLAYS, timeout=TIMEOUT
                            )
                            res = adaptive_benchmark(map(get_rt_from_replay_res, it), logger=logger)
                        except InputGenError as e:
                            res = get_invalid_abr()
                    else:
                        res = get_zero_rt_abr()
                    rtss.append(res)
                    i += 1

        assert len(rtss) == num_inputs and i == num_inputs

        return rtss

    def get_udr_runtimes(ufr: UnrollFactorResult, maybe_non_zero_runtime=None):
        logger.debug(f"Getting runtime for ufr action {ufr.action}, factor {ufr.factor}")
        if ufr.action or ufr.factor == 1:
            try:
                return UnrollFactorRuntimes(
                    ufr.factor, get_module_runtimes(ufr.module, maybe_non_zero_runtime)
                )
            except InputGenError as e:
                logger.debug(e)
        return UnrollFactorRuntimes(ufr.factor, [get_invalid_abr() for _ in range(num_inputs)])

    def get_ud_raw_sample(ud: UnrollDecision):
        x = ud.features
        factor_runtimes = [None for _ in range(ADVICE_TENSOR_LEN)]
        maybe_non_zero_runtime = np.ones(num_inputs, dtype=bool)
        for ufr in ud.results:
            if ufr.factor != 1:
                ufrt = get_udr_runtimes(ufr, maybe_non_zero_runtime)
                assert ufrt.factor >= 2
                factor_runtimes[ufrt.factor - UNROLL_FACTOR_OFFSET] = ufrt.benchmarking_results
                this_maybe_non_zero_runtime = get_maybe_non_zero_runtime_array(
                    ufrt.benchmarking_results
                )
                assert (
                    maybe_non_zero_runtime.shape == this_maybe_non_zero_runtime.shape
                    and maybe_non_zero_runtime.dtype == bool
                    and this_maybe_non_zero_runtime.dtype == bool
                )
                maybe_non_zero_runtime &= this_maybe_non_zero_runtime
                if all(~maybe_non_zero_runtime):
                    logger.debug("Failed to obtain non-0 runtime")
                    return None

        logger.debug(f"Got factor_runtimes {factor_runtimes}")

        # If none of the factors succeeded.
        if all(factor_runtime is None for factor_runtime in factor_runtimes):
            logger.debug("Failed to obtain runtime for any factor")
            return None

        # If we have any factor runtime to compare to, also get the base runtime
        base_runtime = None
        for ufr in ud.results:
            if ufr.factor == 1:
                ufrt = get_udr_runtimes(ufr, maybe_non_zero_runtime)
                base_runtime = ufrt.benchmarking_results
                break
        if base_runtime is None:
            logger.debug("Failed to obtain runtime for base")
            return None

        logger.debug(f"Got base_runtime {base_runtime}")

        return UnrollDecisionRawSample(x, base_runtime, factor_runtimes)

    raw_samples = filter_none(map(get_ud_raw_sample, decision_results))
    if raw:
        yield from raw_samples
    else:
        yield from filter_none(map(get_ud_sample_from_raw, raw_samples))


if __name__ == "__main__":
    parse_args_and_run()
