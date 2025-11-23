# Code adapted from:
# TensorFlow Tutorials – Time Series Forecasting
# https://www.tensorflow.org/tutorials/structured_data/time_series#baselines

import tensorflow as tf

class Baseline(tf.keras.Model):
    def __init__(self, target_cols=None):
        super().__init__()

    def call(self, gt_history_batch):
        # last GT = prediction
        return gt_history_batch[:, -1, :]  
