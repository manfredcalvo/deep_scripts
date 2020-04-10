from tensorflow.python.keras import models
from tensorflow.python.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense, Lambda, Input, GlobalAveragePooling2D


class ImageModel(models.Model):
    def __init__(self, input_data_shape, num_classes, model_name='resnet50', trainable_layers_amount=1):
        super(ImageModel, self).__init__()

        self.num_classes = num_classes
        self.model_name = model_name
        self.trainable_layers_amount = trainable_layers_amount
        self.input_data_shape = input_data_shape

        if self.model_name == 'resnet50':
            self.base_model = ResNet50(include_top=False,
                                       weights='imagenet',
                                       input_tensor=None,
                                       input_shape=self.input_data_shape)
            # Avoid training layers in resnet model.
            layers = self.base_model.layers
            print("Layers name")
            for layer in layers:
                print(layer.name)
                layer.trainable = False
            print("Making layers trainable")
            for layer in layers[-trainable_layers_amount:]:
                print(layer.name)
                layer.trainable = True

        x0 = Input(shape=self.input_data_shape)
        x1 = Lambda(preprocess_input, output_shape=self.input_data_shape)(x0)
        x2 = self.base_model(x1)
        x3 = GlobalAveragePooling2D()(x2)
        x4 = Dense(1024, activation='relu')(x3)
        x5 = Dense(num_classes, activation='softmax', name='softmax')(x4)
        self.model = Model(inputs=x0, outputs=x5)

    def call(self, inputs):
        return self.model(inputs)

    def get_config(self):
        config = {'input_data_shape': self.input_data_shape, 'num_classes': self.num_classes,
                  'model_name': self.model_name,
                  'trainable_layers_amount': self.trainable_layers_amount}
        return config
