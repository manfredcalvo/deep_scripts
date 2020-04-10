
from tensorflow.python.keras import layers
from tensorflow.python.keras.applications.resnet50 import preprocess_input

class PreprocessImage(layers.Layer):

    def __init__(self, model_name = 'resnet_50', **kwargs):
        super(PreprocessImage, self).__init__(**kwargs)
        self.model_name = model_name

    def build(self, input_shape):
        super(PreprocessImage, self).build(input_shape)  # Be sure to call this at the end

    def call(self, input_batch):

        return preprocess_input(input_batch)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        return {'model_name': self.model_name}