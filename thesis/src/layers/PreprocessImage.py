from tensorflow.keras import layers
from tensorflow.keras.utils import get_custom_objects
import tensorflow as tf


class PreprocessImage(layers.Layer):
    def __init__(self, model_name='resnet_50', **kwargs):
        super(PreprocessImage, self).__init__(**kwargs)
        self.model_name = model_name

    def build(self, input_shape):
        super(PreprocessImage, self).build(input_shape)  # Be sure to call this at the end

    def call(self, input_batch):
        if 'resnet_50' in self.model_name:
            return tf.keras.applications.resnet50.preprocess_input(input_batch)
        else:
            if self.model_name == 'mobile_net':
                return tf.keras.applications.mobilenet.preprocess_input(input_batch)
            else:
                if self.model_name == 'resnet_34':
                    return input_batch
                else:
                    return input_batch / 255.0

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        return {'model_name': self.model_name}


get_custom_objects().update({'PreprocessImage': PreprocessImage})
