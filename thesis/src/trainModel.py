import sys
import os
import json
import argparse
from models.ImageModel import ImageModel
from dataset.generate_dataset import GenerateDataset
from dataset.ImageDatasetGenerator import ImageDatasetGenerator
from layers.PreprocessImage import PreprocessImage
from layers.UnsharpMask import UnsharpMask
from layers.UnsharpMaskLog import UnsharpMaskLoG
from optimizers.LearningRateMultiplier import LearningRateMultiplier
from tensorflow.python.keras.applications.resnet50 import ResNet50
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.utils import multi_gpu_model
from tensorflow.python.keras.layers import Dense, Lambda, Input, GlobalAveragePooling2D
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras import Model
from tensorflow.python.keras.callbacks import ReduceLROnPlateau, EarlyStopping, CSVLogger, ModelCheckpoint, \
    LambdaCallback

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# The GPU id to use, usually either "0" or "1"

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def print_layer_weights(layer):
    print()
    print('Actual values of parameters in layer: %s' % layer.name)
    print(layer.get_weights())


def normalization_function(x):
    from tensorflow.python.keras.applications.resnet50 import preprocess_input
    return preprocess_input(x)


def get_model(input_shape, input_layer, num_classes, model_name='resnet_50', trainable_layers_amount=1):
    if model_name == 'resnet_50':
        base_model = ResNet50(include_top=False,
                              weights='imagenet',
                              input_tensor=None,
                              input_shape=input_shape)
    # Avoid training layers in resnet model.
    layers = base_model.layers
    print("Layers name")
    for layer in layers:
        print(layer.name)
        layer.trainable = False
    print("Making layers trainable")
    for layer in layers[-trainable_layers_amount:]:
        print(layer.name)
        layer.trainable = True

    x1 = PreprocessImage(model_name)(input_layer)
    x2 = base_model(x1)
    x3 = GlobalAveragePooling2D()(x2)
    x4 = Dense(1024, activation='relu')(x3)
    output_layer = Dense(num_classes, activation='softmax', name='softmax')(x4)
    return output_layer


if __name__ == '__main__':

    batch_size = 32

    output_dim = 224, 224

    train_percentage = 0.8

    dataset_path = sys.argv[1]

    metadata_path = sys.argv[2]

    with open(metadata_path) as json_file:
        dataset_metadata = json.load(json_file)

    generate_dataset = GenerateDataset(dataset_path, dataset_metadata)

    generate_dataset.load_dataset('None', train_percentage)

    dataset = generate_dataset.all_dataset()

    train_data = dataset['train']
    test_data = dataset['test']

    num_classes = generate_dataset.num_classes()

    print("Num classes: %s" % num_classes)

    image_generator = ImageDatasetGenerator(train_data['files'], train_data['labels'], batch_size, output_dim,
                                            dataset_metadata, True)

    test_image_generator = ImageDatasetGenerator(test_data['files'], test_data['labels'], batch_size, output_dim,
                                                 dataset_metadata, False)

    # image_model = ImageModel((output_dim[0], output_dim[1], 3), num_classes=num_classes, trainable_layers_amount=0)


    input_layer = Input((output_dim[0], output_dim[1], 3))

    unsharp_mask_layer = UnsharpMaskLoG(kernel_size=(5, 5),
                                        regularizer_sigma=None,
                                        regularizer_amount=None,
                                        found_sigma=False,
                                        sigma=1.667)

    #previous_layer = unsharp_mask_layer(input_layer)

    # output_layer = image_model(previous_layer)

    output_layer = get_model((output_dim[0], output_dim[1], 3), input_layer, num_classes=num_classes,
                             trainable_layers_amount=0)

    final_model = Model(inputs=input_layer, outputs=output_layer)

    for layer in final_model.layers:
        print(layer)
        print(layer.name)

    final_model.summary()


    #Barknet 1.0
    #unsharp_mask_multiplier = 10
    #initial_lr = 1e-4
    #BDFR
    #unsharp_mask_multiplier = 1
    #initial_lr = 1e-6
    #Maderas
    unsharp_mask_multiplier = 1
    initial_lr = 1e-6

    optimizer = LearningRateMultiplier(Adam, lr_multipliers={'unsharp_mask': unsharp_mask_multiplier}, lr=initial_lr,
                                       decay=1e-4)

    final_model.compile(optimizer=optimizer,
                        loss='categorical_crossentropy', metrics=['accuracy', 'top_k_categorical_accuracy'])

    '''
    
    model_checkpoint = ModelCheckpoint('model-{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss', save_best_only=True,
                                       save_weights_only=True)
        '''

    print_weights = LambdaCallback(
        on_epoch_end=lambda batch, logs: print_layer_weights(unsharp_mask_layer),
        on_epoch_begin=lambda batch, logs: print_layer_weights(unsharp_mask_layer))

    #Barknet 1.0
    #min_lr = 1e-10
    #BDFR
    min_lr = 1e-12

    callbacks = [ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=min_lr, verbose=1),
                 EarlyStopping(monitor='val_loss', patience=15, verbose=1),
                 CSVLogger('training.log')]

    final_model.fit_generator(image_generator,
                              steps_per_epoch=len(image_generator) // batch_size,
                              validation_data=test_image_generator,
                              validation_steps=len(test_image_generator) // batch_size,
                              epochs=1000,
                              verbose=1,
                              workers=100,
                              use_multiprocessing=True,
                              callbacks=callbacks)
