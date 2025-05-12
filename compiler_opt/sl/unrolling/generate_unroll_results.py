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


from input_gen.utils import Input, InputGenError
from com_pile_utils.dataset_writer import DatasetWriter, ProcessResult
from com_pile_utils.dataset_reader import DatasetReader

from . import unrolling_runner
from .unrolling_runner import UnrollDecisionResult, UnrollDecision
from .unroll_model import ADVICE_TENSOR_LEN, UNROLL_FACTOR_OFFSET, MAX_UNROLL_FACTOR
from input_gen.utils import InputGenReplay, Input

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


PHYSICAL_CORE_RESOURCE = "physical_core"


def main(args):
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    with DatasetReader(args.dataset) as dr:
        if args.one is None:
            physical_cores = psutil.cpu_count(logical=False)
            context = ray.init(log_to_driver=False, resources={PHYSICAL_CORE_RESOURCE: physical_cores - 1})

            args.cpu_mapping = ray.get(get_physical_cpu_mapping.remote())
            logger.info(f"Obtained CPU mapping: {args.cpu_mapping}")
            assert physical_cores == len(args.cpu_mapping)

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


@dataclasses.dataclass(frozen=True)
class UnrollDecisionRuntime:
    factor: int
    runtime: Optional[np.array]


@ray.remote(num_cpus=0, resources={PHYSICAL_CORE_RESOURCE: 1})
def get_physical_cpu_mapping():
    return get_physical_cores()



@ray.remote(num_cpus=0, resources={PHYSICAL_CORE_RESOURCE: 1})
def process_module_wrapper(args, i, data):
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    logger.info(f"cpus {args.cpu_mapping}")
    core_id = ray.get_runtime_context().worker.core_worker.resource_ids()[PHYSICAL_CORE_RESOURCE][0][0]
    assert core_id < len(args.cpu_mapping)
    os.sched_setaffinity(0, [args.cpu_mapping[core_id]])

    return process_module(args, i, data)


def process_module(args, i, data):
    res = process_module_impl_results(args, i, data)
    if res is None:
        return ProcessResult(i, None)
    else:
        assert len(res) == 1
        return process_module_impl_sample(args, i, res[0])


def process_module_impl_results(args, idx, data):
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


def process_module_impl_sample(args, i, data):
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

    samples = list(generate_samples(decision_results, inputs, replay_options, raw=True))

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


def adaptive_benchmark(
    iterator,
    initial_samples=5,
    max_initial_samples=20,
    max_samples=100,
    confidence=0.95,
    relative_ci_threshold=0.05,
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
        mean_estimate, ci_half_width, num_samples
    """

    logger.debug("Starting adaptive benchmarking")

    samples = np.array([], dtype=float)

    n = 0
    while len(samples) < initial_samples and n < max_initial_samples:
        n += 1
        new_sample = next(iterator)
        if new_sample is not None:
            if new_sample == 0:
                logger.debug(f"Got zero")
                return 0
            samples = np.append(samples, new_sample)
            logger.debug(f"Obtained sample {new_sample}, len {len(samples)}")

    if len(samples) < initial_samples:
        logger.debug(f"Too many replay failures")
        return None

    while n < max_samples:
        sample_mean = np.mean(samples)
        sample_std = np.std(samples, ddof=1)  # sample std (unbiased)

        # t critical value
        alpha = 1 - confidence
        t_crit = stats.t.ppf(1 - alpha / 2, df=n - 1)

        margin_error = t_crit * (sample_std / np.sqrt(n))
        relative_ci_width = (2 * margin_error) / sample_mean

        # Check stopping criterion
        if relative_ci_width < relative_ci_threshold:
            logger.debug(f"Converged: mean {sample_mean}, std {sample_std}")
            return sample_mean

        # Get another sample
        new_sample = None
        while new_sample is None and n < max_samples:
            new_sample = next(iterator)
            n += 1
        if new_sample is not None:
            samples = np.append(samples, new_sample)

    logger.debug(f"Could not converge: mean {sample_mean}, std {sample_std}")
    return None

    # final_mean = np.mean(samples)
    # final_margin_error = stats.t.ppf(1 - (1 - confidence)/2, df=n-1) * (np.std(samples, ddof=1) / np.sqrt(n))
    # return final_mean, final_margin_error, n


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


def get_ud_sample_from_raw(x, base_runtime, factor_runtimes):
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

        return (x, np.array(y))


def generate_samples(decision_results, inputs, replay_options, raw=False):
    def get_module_runtimes(module, is_non_zero_runtime=None):
        NUM_REPLAYS = None
        TIMEOUT = 1

        num_inputs = 0
        for entry_inputs in inputs:
            num_inputs += len(entry_inputs)

        rts = np.zeros(num_inputs, dtype=float)
        i = 0
        with InputGenReplay(module, **replay_options) as igr:
            logger.debug(f"Starting replaying {len(inputs)} entries")
            for entry_no, entry_inputs in enumerate(inputs):
                logger.debug(
                    f"Starting replaying for entry {entry_no} with {len(entry_inputs)} inputs"
                )
                for input_no, inpt in enumerate(entry_inputs):
                    if is_non_zero_runtime is None or is_non_zero_runtime[i]:
                        try:
                            it = igr.replay_input(
                                inpt.data, inpt.entry_no, NUM_REPLAYS, timeout=TIMEOUT
                            )
                            res = adaptive_benchmark(map(get_rt_from_replay_res, it))
                        except InputGenError as e:
                            res = None
                        if res is None:
                            rts[i] = np.nan
                        else:
                            rts[i] = res
                    else:
                        rts[i] = 0
                    i += 1

        assert i == num_inputs

        return rts

    def get_udr_runtime(udr: UnrollDecisionResult, is_non_zero_runtime=None):
        logger.debug(f"Getting runtime for udr action {udr.action}, factor {udr.factor}")
        if udr.action or udr.factor == 1:
            try:
                return UnrollDecisionRuntime(
                    udr.factor, get_module_runtimes(udr.module, is_non_zero_runtime)
                )
            except InputGenError as e:
                logger.debug(e)
        return UnrollDecisionRuntime(udr.factor, None)

    def get_ud_sample(ud: UnrollDecision):
        res = get_ud_raw_sample(ud)
        if res is None:
            logger.debug(f"Failed to obtain valid raw sample")
            return None
        x, base_runtime, factor_runtimes = res
        return get_ud_sample_from_raw(x, base_runtime, factor_runtimes)

    def get_ud_raw_sample(ud: UnrollDecision):
        x = ud.features
        factor_runtimes = [None for _ in range(ADVICE_TENSOR_LEN)]
        is_non_zero_runtime = None
        for udr in ud.results:
            if udr.factor != 1:
                udrt = get_udr_runtime(udr, is_non_zero_runtime)
                assert udrt.factor >= 2
                factor_runtimes[udrt.factor - UNROLL_FACTOR_OFFSET] = udrt.runtime
                if is_non_zero_runtime is None:
                    if udrt.runtime is not None:
                        is_non_zero_runtime = udrt.runtime.astype(bool)
                else:
                    is_non_zero_runtime = is_non_zero_runtime & udrt.runtime.astype(bool)
                if is_non_zero_runtime is not None and all(~is_non_zero_runtime):
                    logger.debug("Failed to obtain non-0 runtime")
                    return None

        logging.debug(f"Got factor_runtimes {factor_runtimes}")

        # If none of the factors succeeded.
        if all(factor_runtime is None for factor_runtime in factor_runtimes):
            logger.debug("Failed to obtain runtime for any factor")
            return None

        # If we have any factor runtime to compare to, also get the base runtime
        base_runtime = None
        for udr in ud.results:
            if udr.factor == 1:
                udrt = get_udr_runtime(udr, is_non_zero_runtime)
                base_runtime = udrt.runtime
                if base_runtime is None:
                    logger.debug("Failed to obtain runtime for base")
                    return None

        logging.debug(f"Got base_runtime {base_runtime}")

        return x, base_runtime, factor_runtimes

    def get_ud_samples(uds: Iterable[UnrollDecision]):
        for ud in uds:
            sample = get_ud_sample(ud)
            if sample is not None:
                yield sample
            else:
                logger.debug(f"Failed to obtain sample")

    yield from get_ud_samples(decision_results)


if __name__ == "__main__":
    parse_args_and_run()
