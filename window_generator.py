# Code adapted from:
# TensorFlow Tutorials – Time Series Forecasting
# https://www.tensorflow.org/tutorials/structured_data/time_series#baselines

import numpy as np
import tensorflow as tf


class WindowGenerator:

    def __init__(self, input_width, target_width, shift,
                 input_train_df, input_val_df, input_test_df,
                 target_train_df, target_val_df, target_test_df,
                 target_cols, input_feature_cols, batch_size
                ):

        self.input_width = input_width
        self.target_width = target_width
        self.shift = shift
        self.batch_size = batch_size

        self.target_cols = target_cols
        self.input_feature_cols = input_feature_cols

        self.num_input_features = len(self.input_feature_cols)
        self.num_targets = len(self.target_cols)
        self.total_window_size = self.input_width + self.shift

        self.input_slice = slice(0, self.input_width)
        self.target_index = self.total_window_size - 1

        self.input_train_df = input_train_df
        self.input_val_df = input_val_df
        self.input_test_df = input_test_df

        self.target_train_df = target_train_df
        self.target_val_df = target_val_df
        self.target_test_df = target_test_df

    def _make_seq_dataset(self, df, num_features):
        data = np.array(df, dtype=np.float32)

        ds = tf.keras.utils.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            batch_size=self.batch_size
        )
        return ds  # (B, total_window_size, num_features)

    def split_window(self, inputs_full, targets_full):
        # inputs_full:  (B, total_window_size, num_input_features)
        # targets_full: (B, total_window_size, num_targets)

        X = inputs_full[:, self.input_slice, :]
        X.set_shape([None, self.input_width, self.num_input_features])

        # Final GT step is predicted
        y = targets_full[:, self.target_index, :]
        y.set_shape([None, self.num_targets])

        return X, y

    def make_dataset(self, input_df, target_df, training=False):
        ds_inputs = self._make_seq_dataset(input_df, self.num_input_features)
        ds_targets = self._make_seq_dataset(target_df, self.num_targets)

        # (inputs, targets)
        ds = tf.data.Dataset.zip((ds_inputs, ds_targets))
        ds = ds.map(self.split_window)

        return ds

    def make_baseline_dataset(self, target_df):
        ds = self._make_seq_dataset(target_df, self.num_targets)
        return ds.map(
            lambda batch: (
                # Inputs
                batch[:, :self.input_width, :],
                # Targets
                batch[:, self.target_index, :]
            )
        )


    @property
    def train(self):
        return self.make_dataset(self.input_train_df, self.target_train_df, training=True)

    @property
    def val(self):
        return self.make_dataset(self.input_val_df, self.target_val_df, training=False)

    @property
    def test(self):
        return self.make_dataset(self.input_test_df, self.target_test_df, training=False)

    @property
    def baseline_train(self):
        return self.make_baseline_dataset(self.target_train_df)

    @property
    def baseline_val(self):
        return self.make_baseline_dataset(self.target_val_df)

    @property
    def baseline_test(self):
        return self.make_baseline_dataset(self.target_test_df)
