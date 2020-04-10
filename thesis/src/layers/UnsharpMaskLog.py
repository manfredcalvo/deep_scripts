
import tensorflow as tf
import numpy as np
from tensorflow.python.keras import layers
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.initializers import RandomUniform, Constant
from tensorflow.python.keras.constraints import NonNeg

class UnsharpMaskLoG(layers.Layer):

    def __init__(self,
                 kernel_size,
                 regularizer_sigma,
                 regularizer_amount,
                 found_sigma=False,
                 sigma=None, **kwargs):

        self.kernel_size = kernel_size
        self.regularizer_sigma = regularizer_sigma
        self.regularizer_amount = regularizer_amount
        self.found_sigma = found_sigma
        self.sigma = sigma

        super(UnsharpMaskLoG, self).__init__(**kwargs)

    def build(self, input_shape):

        if self.found_sigma:
            initializer_sigma = RandomUniform(minval=0.0001, maxval=1, seed=None)
            self.sigma = self.add_weight(name='sigma',
                                         shape=(1, 1),
                                         initializer=initializer_sigma,
                                         regularizer=self.regularizer_sigma,
                                         constraint=NonNeg(),
                                         trainable=self.found_sigma)

        initializer_amount = RandomUniform(minval=0, maxval=1, seed=None)

        self.amount = self.add_weight(name='amount',
                                      shape=(1, 1),
                                      initializer=initializer_amount,
                                      regularizer=self.regularizer_amount,
                                      constraint=None,
                                      trainable=True)

        super(UnsharpMaskLoG, self).build(input_shape)

    def LoG_np(self, k, sigma):
        ax = np.round(np.linspace(-np.floor(k / 2), np.floor(k / 2), k))
        x, y = np.meshgrid(ax, ax)
        x2 = np.power(x, 2)
        y2 = np.power(y, 2)
        s2 = np.power(sigma, 2)
        s4 = np.power(sigma, 4)
        hg = np.exp(-(x2 + y2) / (2. * s2))
        kernel_t = hg * (x2 + y2 - 2 * s2) / (s4 * np.sum(hg))
        kernel = kernel_t - np.sum(kernel_t) / np.power(k, 2)
        return kernel

    def call(self, input):

        kernel = self.LoG_np(self.kernel_size[0], self.sigma)

        kernel = K.constant(kernel)

        kernel = K.expand_dims(kernel, axis=-1)
        kernel = K.repeat_elements(kernel, 3, axis=-1)
        kernel = K.expand_dims(kernel)

        B = tf.nn.depthwise_conv2d(input, kernel, (1, 1, 1, 1), padding='SAME')

        U = input + self.amount * B

        maxB = K.max(K.abs(B))

        maxInput = K.max(input)

        U = U * maxInput/maxB

        return U

    def compute_output_shape(self, input_shape):
        return input_shape