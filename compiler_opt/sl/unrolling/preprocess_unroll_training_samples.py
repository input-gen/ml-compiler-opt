import argparse
import logging
import ray

from com_pile_utils.dataset_writer import ProcessResult, ID_FIELD
from com_pile_utils import generate_main
from . import generate_unroll_training_samples
from .datastructures import *
from . import train

logger = logging.getLogger(__name__)


def parse_args_and_run():
    parser = argparse.ArgumentParser(description="Reading ComPileLoop")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dataset", required=True)
    parser.add_argument("--output-parquet", required=True)

    parser.add_argument(
        "--confidence", type=float, default=generate_unroll_training_samples.CONFIDENCE
    )
    parser.add_argument(
        "--relative-ci-threshold-per-sample",
        type=float,
        default=generate_unroll_training_samples.RELATIVE_CI_THRESHOLD * 2,
    )
    parser.add_argument(
        "--relative-ci-threshold-mean",
        type=float,
        default=generate_unroll_training_samples.RELATIVE_CI_THRESHOLD,
    )
    parser.add_argument("--weighted", default=False, action="store_true")
    parser.add_argument(
        "--low-runtime-cutoff", type=float, default=generate_unroll_training_samples.LOW_RUNTIME_CUTOFF
    )

    parser.add_argument("--one", type=int, default=None)
    parser.add_argument("--debug", default=False, action="store_true")

    args = parser.parse_args()
    generate_main.main(args, process_module_wrapper, process_module)

    df = train.get_df(args.output_dataset, remote=True)
    df.info()
    df.to_parquet(args.output_parquet)


@ray.remote
def process_module_wrapper(args, idx, data):
    return process_module(args, idx, data)


def process_module(args, idx, data):
    samples = []
    for sample in data["samples"]:
        assert isinstance(sample, UnrollDecisionRawSample)
        sample = generate_unroll_training_samples.get_ud_sample_from_raw(
            sample,
            confidence=args.confidence,
            relative_ci_threshold_per_sample=args.relative_ci_threshold_per_sample,
            relative_ci_threshold_mean=args.relative_ci_threshold_mean,
            weighted=args.weighted,
        )
        if samples is not None:
            samples.append(sample)
    data["samples"] = samples
    del data[ID_FIELD]
    return ProcessResult(idx, [data])


if __name__ == "__main__":
    parse_args_and_run()
