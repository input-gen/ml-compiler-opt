import argparse
import tensorflow as tf
import pandas as pd
from unroll_model import ADVICE_TENSOR_LEN, UNROLL_FACTOR_OFFSET, MAX_UNROLL_FACTOR

parser = argparse.ArgumentParser()
parser.add_argument('--data', required=True)
args = parser.parse_args()

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

model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(128),
  tf.keras.layers.Dense(128),
  tf.keras.layers.Dense(ADVICE_TENSOR_LEN)
])

model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=tf.keras.optimizers.Adam())
model.fit(unroll_features, unroll_labels, epochs=10)
