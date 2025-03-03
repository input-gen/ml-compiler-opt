import tensorflow as tf
import pandas as pd

unroll_df = pd.read_csv("mock_data.csv")

unroll_features = unroll_df[["feature_1","feature_2"]]
unroll_labels = unroll_df[["unrolling_speedup_1","unrolling_speedup_2"]]

model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(128),
  tf.keras.layers.Dense(128),
  tf.keras.layers.Dense(2)
])

model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=tf.keras.optimizers.Adam())
model.fit(unroll_features, unroll_labels, epochs=10)
