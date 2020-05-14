import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras.initializers import RandomUniform, Constant
from tensorflow.keras.constraints import NonNeg
from tensorflow.keras.utils import get_custom_objects


class UnsharpMask(layers.Layer):
    def __init__(self,
                 kernel_size,
                 found_sigma=True,
                 sigma=None,
                 **kwargs):
        self.kernel_size = kernel_size
        self.found_sigma = found_sigma
        self.sigma = sigma
        super(UnsharpMask, self).__init__(**kwargs)

    def build(self, input_shape):
        # Defining parameters to optimize.

        if self.found_sigma:
            initializer = RandomUniform(minval=0.0001, maxval=1, seed=None)
            # initializer = Constant(0.84089642)
            self.sigma = self.add_weight(name='sigma',
                                         shape=(1, 1),
                                         initializer=initializer,
                                         constraint=NonNeg(),
                                         trainable=True)

        initializer = Constant(0)
        # initializer = Constant(10)
        self.amount = self.add_weight(name='amount',
                                      shape=(1, 1),
                                      initializer=initializer,
                                      trainable=True)

        super(UnsharpMask, self).build(input_shape)  # Be sure to call this at the end

    def call(self, input_batch):

        m, n = [(ss - 1.) / 2. for ss in self.kernel_size]

        y_distance, x_distance = np.ogrid[-m:m + 1, -n:n + 1]

        y_distance = K.constant(y_distance)

        x_distance = K.constant(x_distance)

        result = (K.square(x_distance) + K.square(y_distance)) / (2.0 * K.square(self.sigma))
        result = K.exp(-1 * result)
        result = result / K.sum(result)
        result = K.expand_dims(result, axis=-1)
        result = K.repeat_elements(result, 3, axis=-1)
        result = K.expand_dims(result)

        conv_result = tf.nn.depthwise_conv2d(input_batch, result, (1, 1, 1, 1), padding='SAME')
        D = (input_batch - conv_result)
        V = input_batch + self.amount[0] * D
        V = K.clip(V, 0, 255)
        return V

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        # config = super(UnsharpMask, self).get_config()
        config = {'kernel_size': self.kernel_size,
                  'found_sigma': self.found_sigma,
                  'sigma': self.sigma}
        return config


get_custom_objects().update({'UnsharpMask': UnsharpMask})
