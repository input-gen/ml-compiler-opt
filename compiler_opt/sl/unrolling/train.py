import argparse
import tensorflow as tf
import pandas as pd
import logging
import numpy
import pandas
from statistics import geometric_mean
from datasets import load_dataset

import unrolling_runner

from unroll_model import ADVICE_TENSOR_LEN, UNROLL_FACTOR_OFFSET, MAX_UNROLL_FACTOR
from com_pile_utils.dataset_reader import DatasetReader

logger = logging.getLogger(__name__)


def eval_speedup(model, features, labels):
    oracle_speedups = []
    predicted_speedups = []
    for predicted, (row_idx, real) in zip(model.predict(features), labels.iterrows()):
        logger.debug(f"predicted: {predicted}")
        logger.debug(f"real: {real}")

        oracle_speedup = max(1, real.max())

        if max(predicted) > 1:
            argmax = numpy.argmax(predicted)
            predicted_speedup = real[argmax]
            # We chose an illegal factor
            if predicted_speedup == 0:
                predicted_speedup = 1
        else:
            # We predicted we should not unroll
            predicted_speedup = 1

        logger.debug(predicted_speedup)
        logger.debug(oracle_speedup)
        assert oracle_speedup >= predicted_speedup

        oracle_speedups.append(oracle_speedup)
        predicted_speedups.append(predicted_speedup)

    logger.debug(oracle_speedups)
    logger.debug(predicted_speedups)
    logger.info(f"Geomean oracle speedup: {geometric_mean(oracle_speedups)}")
    logger.info(f"Geomean predicted speedup: {geometric_mean(predicted_speedups)}")


def convert_data_to_df(data):
    features_spec = data["features_spec"]
    advice_spec = data["advice_spec"]
    raw_samples = data["samples"]

    samples = []
    for x, base_rt, factor_rts in raw_samples:
        res = unrolling_runner.get_ud_sample_from_raw(x, base_rt, factor_rts)
        if res is None:
            continue
        samples.append(res)

    flattened_samples = []
    for features, advice in samples:
        flattened_features = []
        for feature in features:
            assert len(feature) == 1
            flattened_features.append(feature[0])
        flattened_samples.append(flattened_features + advice)

    labels = []
    for s in features_spec:
        labels.append(s.name)
    labels += [advice_spec.name + str(i + 2) for i in range(advice_spec.shape[0])]

    df = pandas.DataFrame(flattened_samples, columns=labels)
    logger.debug("Intermediate df")
    logger.debug(df)
    return df


def parse_args_and_run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--debug", default=False, action="store_true")
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    with DatasetReader(args.dataset) as dr:
        dfs = [d for _, d in dr.get_iter()]
        dfs = [convert_data_to_df(df) for df in dfs]
        unroll_df = pd.concat(dfs)
        unroll_df = unroll_df.reset_index(drop=True).astype(float)
        print(unroll_df)

    print(unroll_df.columns)

    cols = list(unroll_df.columns)
    for i, col in enumerate(cols):
        if "unrolling_decision" in col:
            break
    unroll_features = unroll_df[cols[:i]]
    unroll_labels = unroll_df[cols[i:]]
    print(unroll_features.columns)
    print(unroll_labels.columns)

    assert len(unroll_labels.columns) == ADVICE_TENSOR_LEN

    split = len(unroll_features.index) // 2

    print(len(unroll_labels.index))

    test_unroll_features = unroll_features.iloc[split:]
    test_unroll_labels = unroll_labels.iloc[split:]
    print(len(test_unroll_labels.index))

    train_unroll_features = unroll_features.iloc[:split]
    train_unroll_labels = unroll_labels.iloc[:split]
    print(len(train_unroll_labels.index))

    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Dense(128),
            tf.keras.layers.Dense(128),
            tf.keras.layers.Dense(ADVICE_TENSOR_LEN),
        ]
    )

    model.compile(
        loss=tf.keras.losses.MeanSquaredError(),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=[
            tf.keras.metrics.TopKCategoricalAccuracy(i, name="top" + str(i)) for i in range(1, 6)
        ],
    )
    model.fit(train_unroll_features, train_unroll_labels, epochs=10)

    eval_speedup(model, test_unroll_features, test_unroll_labels)


if __name__ == "__main__":
    parse_args_and_run()
