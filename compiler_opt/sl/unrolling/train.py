import argparse
import tensorflow as tf
import pandas as pd
import logging
import numpy
from statistics import geometric_mean
from datasets import load_dataset

from unroll_model import ADVICE_TENSOR_LEN, UNROLL_FACTOR_OFFSET, MAX_UNROLL_FACTOR

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('--data', required=True)
parser.add_argument('--debug', default=False, action='store_true')
args = parser.parse_args()

if args.debug:
    logging.basicConfig(level=logging.DEBUG)
else:
    logging.basicConfig(level=logging.INFO)

if True:
    ds = load_dataset(args.data, split='train', streaming=True)
    unroll_df = pd.DataFrame(ds)
    print(unroll_df)
else:
    unroll_df = pd.read_csv(args.data)

print(unroll_df.columns)

cols = list(unroll_df.columns)
for i, col in enumerate(cols):
    if 'unrolling_decision' in col:
        break
unroll_features = unroll_df[cols[:i]]
unroll_labels = unroll_df[cols[i:]]
print(unroll_features.columns)
print(unroll_labels.columns)

assert len(unroll_labels.columns) == ADVICE_TENSOR_LEN


split = len(unroll_features.index) / 2

print(len(unroll_labels.index))

test_unroll_features = unroll_features.loc[split:]
test_unroll_labels = unroll_labels.loc[split:]
print(len(test_unroll_labels.index))

train_unroll_features = unroll_features.loc[:split]
train_unroll_labels = unroll_labels.loc[:split]
print(len(train_unroll_labels.index))


model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(128),
  tf.keras.layers.Dense(128),
  tf.keras.layers.Dense(ADVICE_TENSOR_LEN)
])

model.compile(
    loss=tf.keras.losses.MeanSquaredError(),
    optimizer=tf.keras.optimizers.Adam(),
    metrics=[
        tf.keras.metrics.TopKCategoricalAccuracy(i, name='top' + str(i)) for i in range(1, 6)
    ],
)
model.fit(train_unroll_features, train_unroll_labels, epochs=10)

def eval_speedup(model, features, labels):
    oracle_speedups = []
    predicted_speedups = []
    for predicted, (row_idx, real) in zip(model.predict(features), labels.iterrows()):
        logger.debug(f'predicted: {predicted}')
        logger.debug(f'real: {real}')

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
    logger.info(f'Geomean oracle speedup: {geometric_mean(oracle_speedups)}')
    logger.info(f'Geomean predicted speedup: {geometric_mean(predicted_speedups)}')

eval_speedup(model, test_unroll_features, test_unroll_labels)
