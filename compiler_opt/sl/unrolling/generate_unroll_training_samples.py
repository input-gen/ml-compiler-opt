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
import pandas as pd
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
                log_to_driver=True, resources={PHYSICAL_CORE_RESOURCE: physical_cores - 1}
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
        uch = unrolling_runner.UnrollCompilerHost(True, args.debug)

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
    if sample_mean == 0:
        assert sample_std == 0
        return 0.0, 0.0

    # t critical value
    n = len(samples)
    alpha = 1 - confidence
    t_crit = stats.t.ppf(1 - alpha / 2, df=n - 1)

    margin_error = t_crit * (sample_std / np.sqrt(n))
    relative_ci_width = (2 * margin_error) / sample_mean
    return sample_mean, relative_ci_width


def adaptive_benchmark_baseline(
    iterator,
    logger=logger,
):
    return adaptive_benchmark(
        iterator,
        initial_samples=INITIAL_SAMPLES,
        max_initial_samples=MAX_INITIAL_SAMPLES,
        max_samples=100,
        confidence=0.95,
        relative_ci_threshold=0.02,
        logger=logger,
        fail_on_non_convergence=False,
    )


def adaptive_benchmark_factor(
    iterator,
    logger=logger,
):
    return adaptive_benchmark(
        iterator,
        initial_samples=INITIAL_SAMPLES,
        max_initial_samples=MAX_INITIAL_SAMPLES,
        max_samples=MAX_SAMPLES,
        confidence=CONFIDENCE,
        relative_ci_threshold=RELATIVE_CI_THRESHOLD,
        logger=logger,
        fail_on_non_convergence=False,
    )


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
    # This will get element wise speedup factors for all inputs where both succeeded
    arr = base / opt
    arr = arr[~np.isnan(arr)]  # remove NaNs
    if arr.size == 0:
        return None
    geomean = np.exp(np.mean(np.log(arr)))
    return geomean


def get_benchmarking_stats(samples, confidence):
    remove_ratio_ns = [0, 1, 2, 3, 4]
    remove_ratio_d = 10

    assert len(samples.shape) == 1

    num_samples = len(samples)
    sorted_samples = np.sort(samples)

    for i, remove_ratio_n in enumerate(remove_ratio_ns):
        num_to_remove = round(remove_ratio_n * num_samples / remove_ratio_d)
        if num_to_remove == 0:
            removed_samples = sorted_samples
        else:
            removed_samples = sorted_samples[:-num_to_remove]
        mean, ci = get_benchmarking_mean_ci(removed_samples, confidence)
        if len(removed_samples) > 0:
            med = np.median(removed_samples)
        else:
            med = np.nan
        yield {"mean": mean, "median": med, "ci": ci, "num": num_samples - num_to_remove}


def reduce_abr_min_ci(abr: AdaptiveBenchmarkingResult, confidence):
    it = get_benchmarking_stats(abr.runtimes, confidence)
    return min(list(it), key=lambda x: x["ci"] if not np.isnan(x["ci"]) else np.inf)


def reduce_abr(abr: AdaptiveBenchmarkingResult, confidence, relative_ci_threshold):
    it = get_benchmarking_stats(abr.runtimes, confidence)
    try:
        while True:
            res = next(it)
            if res["ci"] < relative_ci_threshold:
                return res["med"]
    except StopIteration:
        return np.nan


LOW_RUNTIME_CUTOFF = 1000


def get_ud_sample_from_raw(
    udrs,
    confidence=0.95,
    relative_ci_threshold_per_sample=RELATIVE_CI_THRESHOLD * 2,
    relative_ci_threshold_mean=RELATIVE_CI_THRESHOLD,
    weighted=False,
    low_runtime_cutoff=LOW_RUNTIME_CUTOFF,
    logger=logger,
):
    all_ufrts = [udrs.base_ufrts] + udrs.factors_ufrts

    rts = {}
    for i, ufrt in enumerate(all_ufrts):
        assert ufrt.factor == UNROLL_FACTOR_OFFSET + i - 1
        if ufrt.action:
            factor_inputs_rt = list(
                map(
                    lambda x: reduce_abr_min_ci(x, confidence),
                    ufrt.benchmarking_results,
                )
            )
            rts[ufrt.factor] = factor_inputs_rt
    assert len(rts) > 0
    assert 1 in rts.keys()
    frames = {factor: pd.DataFrame(inputs) for factor, inputs in rts.items()}

    # Inputs are the rows, and the factors are the columns.
    df = pd.concat(frames, axis=1)
    df.columns.names = ["factor", "stat"]

    # Drop any inputs where we have zero or nan runtime.
    medians = df.xs("median", axis=1, level="stat")
    mask = (medians == 0) | (medians.isna())
    df = df[~mask.any(axis=1)]
    if len(df) == 0:
        logger.debug(f"Failed to obtain valid runtime for all factors")
        return None

    # Drop any inputs where the base runtime is too low
    medians = df.xs("median", axis=1, level="stat")
    mask = medians[1] < low_runtime_cutoff
    df = df[~mask]

    cis = df.xs("ci", axis=1, level="stat")
    mask_per_sample = (cis < relative_ci_threshold_per_sample).any(axis=1)
    cis_mean = cis.mean(axis=1)
    mask_mean = cis_mean < relative_ci_threshold_mean
    df = df[mask_per_sample & mask_mean]
    if len(df) == 0:
        logger.debug(f"Failed to obtain statistically confident runtime for all factors")
        return None

    # Grab only the medians
    df = df.xs("median", axis=1, level="stat")

    assert not any(df.isna().any(axis=1))

    # Get the speedups relative to factor 1 in each input.
    for factor in rts.keys():
        if factor != 1:
            df[factor] = df[1] / df[factor]

    assert not any(df.isna().any(axis=1))

    # Encode inability to unroll (action = False) as speedup = 0.
    for factor in {i for i in range(UNROLL_FACTOR_OFFSET, MAX_UNROLL_FACTOR + 1)} - rts.keys():
        df[factor] = 0.0

    assert not any(df.isna().any(axis=1))

    # Drop the baseline
    baseline = df[1]
    df.drop(1, axis=1, inplace=True)

    assert not any(df.isna().any(axis=1))

    # Get the geomean speedup for each factor across all inputs.
    if weighted:
        weights = baseline / baseline.sum()
        speedups = df.apply(lambda col: np.exp(np.sum(weights * np.log(col))), axis=0)
    else:
        speedups = df.apply(lambda col: np.exp(np.mean(np.log(col))), axis=0)

    assert not any(speedups == np.nan)

    return UnrollDecisionTrainingSample(udrs.features, udrs.heuristic_factor, speedups)


def filter_none(l):
    return filter(lambda x: x is not None, l)


def get_have_valid_runtime_array(abrs: List[AdaptiveBenchmarkingResult]):
    return np.array(list(map(lambda x: not x.is_invalid(), abrs)), dtype=bool)


def get_have_invalid_runtime_array(abrs: List[AdaptiveBenchmarkingResult]):
    return np.array(list(map(lambda x: x.is_invalid(), abrs)), dtype=bool)


def get_maybe_non_zero_runtime_array(abrs: List[AdaptiveBenchmarkingResult]):
    return np.array(list(map(lambda x: not x.is_zero_rt(), abrs)), dtype=bool)


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

    def get_module_runtimes(module, is_baseline, maybe_non_zero_runtime=None, invalid_runtime=None):
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
                    if maybe_non_zero_runtime is not None and not maybe_non_zero_runtime[i]:
                        res = get_zero_rt_abr()
                    elif invalid_runtime is not None and invalid_runtime[i]:
                        res = get_invalid_abr()
                    else:
                        try:
                            it = igr.replay_input(
                                inpt.data, inpt.entry_no, NUM_REPLAYS, timeout=TIMEOUT
                            )
                            if is_baseline:
                                bm_f = adaptive_benchmark_baseline
                            else:
                                bm_f = adaptive_benchmark_factor
                            res = bm_f(map(get_rt_from_replay_res, it), logger=logger)
                        except InputGenError as e:
                            res = get_invalid_abr()
                    rtss.append(res)
                    i += 1

        assert len(rtss) == num_inputs and i == num_inputs

        return rtss

    def get_udr_runtimes(ufr: UnrollFactorResult, maybe_non_zero_runtime=None, invalid_runtime=None):
        logger.debug(f"Getting runtime for ufr action {ufr.action}, factor {ufr.factor}")
        if ufr.action or ufr.factor == 1:
            try:
                return UnrollFactorRuntimes(
                    ufr.factor,
                    True,
                    get_module_runtimes(
                        ufr.module, ufr.factor == 1, maybe_non_zero_runtime, invalid_runtime
                    ),
                )
            except InputGenError as e:
                logger.debug(e)
                return UnrollFactorRuntimes(
                    ufr.factor, True, [get_invalid_abr() for _ in range(num_inputs)]
                )
        else:
            return UnrollFactorRuntimes(
                ufr.factor, False, [get_invalid_abr() for _ in range(num_inputs)]
            )

    def get_ud_raw_sample(ud: UnrollDecision):
        x = ud.features
        runtimes = [None for _ in range(ADVICE_TENSOR_LEN + 1)]
        # For each input, whether it may be a non-zero runtime input.
        maybe_non_zero_runtime = np.ones(num_inputs, dtype=bool)
        # For each input, whether we have encountered an invalid runtime
        # already. In such cases, we don't need to benchmark any more factors.
        invalid_runtime = np.zeros(num_inputs, dtype=bool)
        # TODO maybe we can use this to give some leeway for subsequent factor
        # benchmarking if we were on the edge of replay timeout.
        # have_valid_runtime = np.zeros(num_inputs, dtype=bool)
        for ufr in ud.results:
            ufrt = get_udr_runtimes(ufr, maybe_non_zero_runtime, invalid_runtime)
            runtimes[ufrt.factor - UNROLL_FACTOR_OFFSET + 1] = ufrt
            this_maybe_non_zero_runtime = get_maybe_non_zero_runtime_array(ufrt.benchmarking_results)
            assert maybe_non_zero_runtime.shape == (len(ufrt.benchmarking_results),)
            assert maybe_non_zero_runtime.shape == this_maybe_non_zero_runtime.shape
            assert maybe_non_zero_runtime.dtype == bool
            assert this_maybe_non_zero_runtime.dtype == bool
            maybe_non_zero_runtime &= this_maybe_non_zero_runtime

            # this_have_valid_runtime = get_have_valid_runtime_array(
            #     ufrt
            # )
            # have_valid_runtime |= this_have_valid_runtime

            if ufrt.action:
                this_have_invalid_runtime = get_have_invalid_runtime_array(ufrt.benchmarking_results)
                invalid_runtime |= this_have_invalid_runtime

            if all(~maybe_non_zero_runtime):
                logger.debug("Failed to obtain non-0 runtime")
                return None
            if all(invalid_runtime):
                logger.debug("Failed to obtain non-0 runtime")
                return None

        logger.debug(f"Got runtimes {runtimes}")

        assert all(rt is not None for rt in runtimes), "Missing unroll factors"

        if all(not rt.action for rt in runtimes[1:]):
            logger.debug("Failed to get action for any factor")
            return None

        assert runtimes[0].factor == 1, "Base does not have unroll factor == 1"
        assert runtimes[0].action, "Action has to be true on the base"

        return UnrollDecisionRawSample(x, ud.heuristic_factor, runtimes[0], runtimes[1:])

    raw_samples = filter_none(map(get_ud_raw_sample, decision_results))
    if raw:
        yield from raw_samples
    else:
        yield from filter_none(map(get_ud_sample_from_raw, raw_samples))


if __name__ == "__main__":
    parse_args_and_run()
