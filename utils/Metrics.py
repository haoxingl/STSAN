import tensorflow as tf
from tensorflow import math


class RMSE(tf.keras.metrics.Mean):
    def __init__(self, name='RMSE', dtype=tf.float32):
        super(RMSE, self).__init__(name, dtype=dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, self._dtype)
        y_pred = tf.cast(y_pred, self._dtype)
        error = math.squared_difference(y_true, y_pred)
        return super(RMSE, self).update_state(error, sample_weight=sample_weight)

    def result(self):
        return math.sqrt(math.divide_no_nan(self.total, self.count))


class MAE(tf.keras.metrics.Mean):
    def __init__(self, name='MAE', dtype=tf.float32):
        super(MAE, self).__init__(name, dtype=dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        error = math.abs(math.subtract(y_true, y_pred))
        return super(MAE, self).update_state(error, sample_weight=sample_weight)

    def result(self):
        return math.divide_no_nan(self.total, self.count)


class MAPE(tf.keras.metrics.Mean):
    def __init__(self, name='MAPE', dtype=tf.float32):
        super(MAPE, self).__init__(name, dtype=dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        error = 100 * math.abs(math.subtract(y_true, y_pred)) / y_true
        return super(MAPE, self).update_state(error, sample_weight=sample_weight)

    def result(self):
        return math.divide_no_nan(self.total, self.count)
