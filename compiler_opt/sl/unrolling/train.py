import argparse
import tensorflow as tf
import pandas as pd
import logging
import numpy as np
import pandas
from statistics import geometric_mean

from .unroll_model import ADVICE_TENSOR_LEN, UNROLL_FACTOR_OFFSET, MAX_UNROLL_FACTOR
from .datastructures import UnrollDecisionRawSample, UnrollDecisionTrainingSample
from com_pile_utils.dataset_reader import DatasetReader
from . import *

logger = logging.getLogger(__name__)


def eval_speedup(model, features, labels):
    oracle, predicted = speedup_metric([v for i, v in labels.iterrows()], model.predict(features))

    logger.info(f"Geomean oracle speedup: {oracle}")
    logger.info(f"Geomean predicted speedup: {predicted}")


def speedup_metric(y_true, y_pred):
    oracle_speedups = []
    predicted_speedups = []
    for predicted, real in zip(y_pred, y_true):
        logger.debug(f"predicted: {predicted}")
        logger.debug(f"real: {real}")

        oracle_speedup = max(1, real.max())

        if max(predicted) > 1:
            argmax = np.argmax(predicted)
            predicted_speedup = real.iloc[argmax]
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

    return geometric_mean(oracle_speedups), geometric_mean(predicted_speedups)


def convert_data_to_df(
    data,
    relative_ci_threshold=generate_unroll_training_samples.RELATIVE_CI_THRESHOLD,
):
    features_spec = data["features_spec"]
    advice_spec = data["advice_spec"]
    samples = data["samples"]

    flattened_samples = []
    for sample in samples:
        if isinstance(sample, UnrollDecisionRawSample):
            sample = generate_unroll_training_samples.get_ud_sample_from_raw(
                sample, relative_ci_threshold
            )
            if sample is None:
                continue
        elif isinstance(sample, UnrollDecisionTrainingSample):
            pass
        else:
            assert False

        flattened_features = np.concatenate(sample.features + [sample.advice])
        flattened_samples.append(flattened_features)

    labels = []
    for s in features_spec:
        assert len(s.shape) == 1
        for i in range(s.shape[0]):
            labels.append(s.name + str(i))
    labels += [advice_spec.name + str(i + 2) for i in range(advice_spec.shape[0])]

    df = pandas.DataFrame(flattened_samples, columns=labels)
    logger.debug("Intermediate df")
    logger.debug(df)
    return df


def get_data(dataset):
    with DatasetReader(dataset) as dr:
        return [d for _, d in dr.get_iter()]


def get_df(
    dataset,
    relative_ci_threshold=generate_unroll_training_samples.RELATIVE_CI_THRESHOLD,
):
    logger.info("Loading dataset...")
    datas = get_data(dataset)
    logger.info("Done.")
    logger.info("Converting data...")
    unroll_df = pd.concat(map(lambda x: convert_data_to_df(x, relative_ci_threshold), datas))
    unroll_df = unroll_df.reset_index(drop=True).astype(float)
    logger.info("Done.")
    return unroll_df


def get_X_Y_from_df(unroll_df):
    cols = list(unroll_df.columns)
    for i, col in enumerate(cols):
        if "unrolling_decision" in col:
            break
    unroll_features = unroll_df[cols[:i]]
    unroll_labels = unroll_df[cols[i:]]
    logger.debug(unroll_features.columns)
    logger.debug(unroll_labels.columns)
    return unroll_features, unroll_labels


def get_X_Y(dataset):
    unroll_df = get_df(dataset)
    logger.info(unroll_df.columns)
    return get_X_Y_from_df(unroll_df)


def split(X, Y, ratio=0.8):
    assert len(Y.columns) == ADVICE_TENSOR_LEN

    split = int(len(X.index) * ratio)

    logger.debug(len(Y.index))

    X_test = X.iloc[split:]
    Y_test = Y.iloc[split:]
    logger.info(f"Testing split: {len(Y_test.index)}")

    X_train = X.iloc[:split]
    Y_train = Y.iloc[:split]
    logger.info(f"Training split: {len(Y_train.index)}")

    return X_train, Y_train, X_test, Y_test


def get_metrics():
    return [tf.keras.metrics.TopKCategoricalAccuracy(i, name="top" + str(i)) for i in range(1, 6)]


class EvalCallback(tf.keras.callbacks.Callback):
    def __init__(self, label, x, y):
        super().__init__()
        self.label = label
        self.x = x
        self.y = y

    def on_epoch_end(self, epoch, logs=None):
        oracle, predicted = speedup_metric(
            [v for i, v in self.y.iterrows()], self.model.predict(self.x)
        )
        logs[self.label + "_oracle_speedup"] = oracle
        logs[self.label + "_predicted_speedup"] = predicted


def parse_args_and_run():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dataset")
    group.add_argument("--preprocessed-dataset")
    parser.add_argument("--preprocessed-output-dataset")
    parser.add_argument("--debug", default=False, action="store_true")
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    if args.preprocessed_dataset is not None and args.preprocessed_output_dataset is not None:
        print("Already preprocessed")
        return 1

    if args.dataset is not None and args.preprocessed_output_dataset is not None:
        df = get_df(args.dataset)
        df.to_parquet(args.preprocessed_output_dataset)
        return 0

    if args.preprocessed_dataset is not None:
        df = pd.read_parquet(args.preprocessed_dataset)
    elif args.dataset is not None:
        df = get_df(args.dataset)
    else:
        assert False

    df = df.sample(frac=1).reset_index(drop=True)
    X, Y = get_X_Y_from_df(df)
    X_train, Y_train, X_test, Y_test = split(X, Y)

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
        metrics=get_metrics(),
    )
    model.fit(
        X_train,
        Y_train,
        epochs=10,
        callbacks=[
            EvalCallback("train", X_train, Y_train),
            EvalCallback("test", X_test, Y_test),
        ],
    )

    eval_speedup(model, X_test, Y_test)

    return 0


if __name__ == "__main__":
    exit(parse_args_and_run())
