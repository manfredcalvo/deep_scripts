from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Model


def freeze_layers(model, layers):
    for i in model.layers:
        if i.trainable:
            i.trainable = False
            layers.append(i)
        if isinstance(i, Model):
            freeze_layers(i, layers)


class CustomModelCheckpoint(ModelCheckpoint):
    def __init__(self, *args, **kwargs):
        super(CustomModelCheckpoint, self).__init__(*args, **kwargs)

    def on_epoch_end(self, epoch, logs=None):
        layers_freezed = []
        freeze_layers(self.model, layers_freezed)
        super(CustomModelCheckpoint, self).on_epoch_end(epoch, logs)
        # Unfreezing layers trainable.
        for layer in layers_freezed:
            layer.trainable = True
