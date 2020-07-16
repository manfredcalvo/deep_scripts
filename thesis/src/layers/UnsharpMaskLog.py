import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras.initializers import RandomUniform, Constant
from tensorflow.keras.constraints import NonNeg
from tensorflow.keras.utils import get_custom_objects


class UnsharpMaskLoG(layers.Layer):
    def __init__(self,
                 kernel_size,
                 found_sigma=False,
                 sigma=1.667,
                 amount=None,
                 **kwargs):
        self.kernel_size = kernel_size
        self.found_sigma = found_sigma
        self.sigma = sigma
        self.amount = amount

        super(UnsharpMaskLoG, self).__init__(**kwargs)

    def build(self, input_shape):
        if self.found_sigma:
            initializer_sigma = RandomUniform(minval=0.0001, maxval=1, seed=None)
            self.sigma = self.add_weight(name='sigma',
                                         shape=(1, 1),
                                         initializer=initializer_sigma,
                                         constraint=NonNeg(),
                                         trainable=self.found_sigma)
        if self.amount:
            print("Initializing amount as a constant value: %s" % self.amount)
            initializer_amount = Constant(self.amount)
        else:
            initializer_amount = RandomUniform(minval=0, maxval=1, seed=None)

        self.amount = self.add_weight(name='amount',
                                      shape=(1, 1),
                                      initializer=initializer_amount,
                                      constraint=None,
                                      trainable=True)

        super(UnsharpMaskLoG, self).build(input_shape)

    def LoG_np_mathlab(self, sigma, kernel_size):
        m, n = [(ss - 1.) / 2. for ss in kernel_size]
        y, x = np.ogrid[-m:m + 1, -n:n + 1]
        x2 = np.power(x, 2)
        y2 = np.power(y, 2)
        s2 = np.power(sigma, 2)
        s6 = np.power(sigma, 6)
        hg = np.exp(-1 * ((x2+y2) / 2 * s2))
        h = (x2 + y2 - 2 * s2) * hg
        h = h / (2 * np.pi * s6 * hg.sum())
        return h
    def LoG_np_new(self, sigma, kernel_size):
        m, n = [(ss - 1.) / 2. for ss in kernel_size]
        y, x = np.ogrid[-m:m + 1, -n:n + 1]
        x2 = np.power(x, 2)
        y2 = np.power(y, 2)
        s2 = np.power(sigma, 2)
        s4 = np.power(sigma, 4)
        hg = np.exp(-(x2 + y2) / (2. * s2))
        kernel_t = hg * (1 - (x2 + y2 / 2 * s2)) * (1.0 / s4 * hg.sum())
        kernel = kernel_t - kernel_t.sum() / np.power(kernel_size[0], 2)
        return kernel

    def call(self, input_batch):
        kernel = self.LoG_np_new(self.sigma, self.kernel_size)

        kernel = K.constant(kernel)
        kernel = K.expand_dims(kernel, axis=-1)
        kernel = K.repeat_elements(kernel, 3, axis=-1)
        kernel = K.expand_dims(kernel)

        B = tf.nn.depthwise_conv2d(input_batch, kernel, (1, 1, 1, 1), padding='SAME')

        U = (input_batch + (self.amount[0] * B))

        U = K.clip(U, 0, 255)

        return U

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {'kernel_size': self.kernel_size,
                  'found_sigma': self.found_sigma,
                  'sigma': self.sigma}
        return config


get_custom_objects().update({'UnsharpMaskLoG': UnsharpMaskLoG})
