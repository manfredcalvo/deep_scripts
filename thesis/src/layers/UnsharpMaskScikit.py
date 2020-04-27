
import tensorflow as tf
import numpy as np
from tensorflow.python.keras import layers
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.initializers import RandomUniform, Constant
from tensorflow.python.keras.constraints import NonNeg
from tensorflow.python.keras.utils import get_custom_objects


class UnsharpMaskScikit(layers.Layer):
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
        super(UnsharpMaskScikit, self).__init__(**kwargs)

    def build(self, input_shape):
        # Defining parameters to optimize.

        if self.found_sigma:

            self.sigma = self.add_weight(name='sigma',
                                         shape=(1, 1),
                                         initializer=Constant(0.2),
                                         regularizer=self.regularizer_sigma,
                                         constraint=NonNeg(),
                                         trainable=True)
            '''
            self.sigma = self.add_weight(name='sigma', 
                                          shape=(1,1),
                                          initializer=RandomUniform(minval=0.0001, maxval=1, seed=None),
                                          regularizer=self.kernel_regularizer_sigma,
                                          constraint = NonNeg(),
                                          trainable=True)
                                          '''
        else:
            self.sigma = K.constant(np.array(self.sigma))

        self.amount = self.add_weight(name='amount',
                                      shape=(1, 1),
                                      initializer=Constant(5),
                                      regularizer=self.regularizer_amount,
                                      constraint=MaxNorm(max_value=100, axis=0),
                                      trainable=True)
        '''
        self.amount = self.add_weight(name='amount', 
                                      shape=(1,1),
                                      initializer=RandomUniform(minval=0, maxval=10, seed=None),
                                      regularizer=self.kernel_regularizer_amount,
                                      constraint = MaxNorm(max_value=100, axis=0),
                                      trainable=True)
        '''
        super(UnsharpMaskScikit, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):

        truncate = 4.0

        kernel_size = K.round(truncate * self.sigma[0] + 0.5)

        dim = K.eval(kernel_size)

        tensor_range = K.arange(dim[0], dtype='float32')

        x_distance = K.concatenate([-1 * tensor_range, K.zeros(1), tensor_range])

        x_shape = K.int_shape(x_distance)

        y_distance = K.reshape(x_distance, (1, x_shape[0]))

        result = (K.square(x_distance) + K.square(y_distance)) / (2.0 * K.square(self.sigma[0]))
        result = K.exp(-1 * result)
        result = result / K.sum(result)

        result = K.expand_dims(result, axis=-1)
        result = K.repeat_elements(result, 3, axis=-1)
        result = K.expand_dims(result)

        conv_result = tf.nn.depthwise_conv2d(x, result, (1, 1, 1, 1), padding='SAME')

        return x + (self.amount[0] * (x - conv_result))

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

get_custom_objects().update({'UnsharpMaskScikit': UnsharpMaskScikit})