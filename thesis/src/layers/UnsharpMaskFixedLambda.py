import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras.initializers import RandomUniform, Constant
from tensorflow.keras.constraints import NonNeg
from tensorflow.keras.utils import get_custom_objects
import cv2


class UnsharpMaskFixedLambda(layers.Layer):
    def __init__(self,
                 kernel_size,
                 found_sigma=True,
                 sigma=None,
                 amount=None,
                 **kwargs):
        self.kernel_size = kernel_size
        self.found_sigma = found_sigma
        self.sigma = sigma
        self.amount = amount
        super(UnsharpMaskFixedLambda, self).__init__(**kwargs)

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

        super(UnsharpMaskFixedLambda, self).build(input_shape)  # Be sure to call this at the end

    def gaussian_kernel_cv2(self, kernel_size, sigma):
        kernel = cv2.getGaussianKernel(kernel_size, sigma)
        kernel = np.dot(kernel, kernel.transpose())
        return kernel

    def gaussian_kernel(self, kernel_size):
        m, n = [(ss - 1.) / 2. for ss in kernel_size]
        y_distance, x_distance = np.ogrid[-m:m + 1, -n:n + 1]
        result = (np.square(x_distance) + np.square(y_distance)) / (2.0 * np.square(self.sigma))
        result = np.exp(-1 * result)
        result = result / result.sum()
        return result

    def reshape_function(self, lambdas):
        (width, height, channels) = 224, 224, 3
        c = K.repeat_elements(lambdas, width, axis=1)
        d = K.expand_dims(c)
        e = K.repeat_elements(d, height, axis=2)
        f = K.expand_dims(e)
        g = K.repeat_elements(f, channels, axis=3)
        return g

    def call(self, inputs):
        lambdas_batch, input_batch = inputs

        kernel = self.gaussian_kernel_cv2(self.kernel_size[0], self.sigma)
        kernel = K.constant(kernel)
        kernel = K.expand_dims(kernel, axis=-1)
        kernel = K.repeat_elements(kernel, 3, axis=-1)
        kernel = K.expand_dims(kernel)

        blurred = tf.nn.depthwise_conv2d(input_batch, kernel, (1, 1, 1, 1), padding='SAME')

        D = (input_batch - blurred)
        lambdas_batch = self.reshape_function(lambdas_batch)
        Z = lambdas_batch * D
        V = (input_batch + Z)
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


get_custom_objects().update({'UnsharpMaskFixedLambda': UnsharpMaskFixedLambda})
