import tensorflow as tf
import numpy as np
from tensorflow.python.keras import layers
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.initializers import RandomUniform
from tensorflow.python.keras.constraints import NonNeg


class UnsharpMask(layers.Layer):
    def __init__(self,
                 kernel_size,
                 regularizer_sigma,
                 regularizer_amount,
                 found_sigma=True,
                 sigma=None,
                 **kwargs):
        self.kernel_size = kernel_size
        self.regularizer_sigma = regularizer_sigma
        self.regularizer_amount = regularizer_amount
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
                                         regularizer=self.regularizer_sigma,
                                         constraint=NonNeg(),
                                         trainable=True)
        else:
            self.sigma = K.constant(self.sigma, shape=(1, 1))

        initializer = RandomUniform(minval=0, maxval=1, seed=None)
        # initializer = Constant(10)
        self.amount = self.add_weight(name='amount',
                                      shape=(1, 1),
                                      initializer=initializer,
                                      regularizer=self.regularizer_amount,
                                      # constraint = MaxNorm(max_value=50, axis=0),
                                      trainable=True)

        super(UnsharpMask, self).build(input_shape)  # Be sure to call this at the end

    def call(self, input_batch):

        m, n = [(ss - 1.) / 2. for ss in self.kernel_size]

        y_distance, x_distance = np.ogrid[-m:m + 1, -n:n + 1]

        y_distance = K.constant(y_distance)

        x_distance = K.constant(x_distance)

        result = (K.square(x_distance) + K.square(y_distance)) / (2.0 * K.square(self.sigma[0]))
        result = K.exp(-1 * result)
        result = result / K.sum(result)
        result = K.expand_dims(result, axis=-1)
        result = K.repeat_elements(result, 3, axis=-1)
        result = K.expand_dims(result)

        conv_result = tf.nn.depthwise_conv2d(input_batch, result, (1, 1, 1, 1), padding='SAME')
        D = (input_batch - conv_result)
        # max_d = K.max(K.abs(D))
        # max_input  = K.max(input_batch)
        V = input_batch + self.amount[0] * D
        # V = V * max_d/max_input
        return V

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        #config = super(UnsharpMask, self).get_config()
        config = {'kernel_size': self.kernel_size,
                       'regularizer_sigma': self.regularizer_sigma,
                       'regularizer_amount': self.regularizer_amount,
                       'found_sigma': self.found_sigma,
                       'sigma': self.sigma}
        return config
