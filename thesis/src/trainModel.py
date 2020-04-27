import os
import json
import argparse
import uuid
from datetime import datetime
import inspect
import pandas as pd
import math
import numpy as np
from dataset.generate_dataset import GenerateDataset
from dataset.ImageDatasetGenerator import ImageDatasetGenerator
from layers.PreprocessImage import PreprocessImage
from layers.UnsharpMaskLog import UnsharpMaskLoG
from layers.UnsharpMask import UnsharpMask
from optimizers.LearningRateMultiplier import LearningRateMultiplier
from tensorflow.python.keras.applications.resnet50 import ResNet50
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.applications.inception_v3 import InceptionV3
from tensorflow.python.keras.applications.vgg16 import VGG16
from tensorflow.python.keras.optimizers import Adam
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


def get_model(input_shape, input_layer, num_classes, model_name='resnet_50', trainable_layers_amount=1):
    if model_name == 'resnet_50':
        base_model = ResNet50(include_top=False,
                              weights='imagenet',
                              input_tensor=None,
                              input_shape=input_shape)
    else:
        if model_name == 'vgg_16':
            base_model = VGG16(include_top=False,
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


def calculate_steps(dataset_size, batch_size):
    return math.ceil(dataset_size / batch_size)


def create_parse():
    parser = argparse.ArgumentParser(
        description='Training deep neuronal network to predict the specie of an specimen represented by an image'
    )

    parser.add_argument('--dataset_path', '-d',
                        required=True,
                        help='Dataset of images')

    parser.add_argument('--metadata_path', '-m',
                        required=True,
                        help='Metadata of the dataset')

    parser.add_argument('--output_experiments', '-o',
                        required=True,
                        help='Folder of the experiments results')

    parser.add_argument('--settings', '-s',
                        required=True,
                        help='Setting of the experiment')

    return parser


# Barknet 1.0
# unsharp_mask_multiplier = 10
# initial_lr = 1e-4
# batch_size = 32
# output_dim = 224, 224
# training_percentage = 0.8
# min_lr = 1e-10
# fixed_sigma = 1.667
# reduce_lr_factor = 0.2
# reduce_lr_patience = 5
# BDFR
# unsharp_mask_multiplier = 1
# initial_lr = 1e-6
# batch_size = 32
# output_dim = 224, 224
# training_percentage = 0.8
# min_lr = 1e-12
# fixed_sigma = 1.667
# reduce_lr_factor = 0.2
# reduce_lr_patience = 5
# Maderas
# unsharp_mask_multiplier = 1
# initial_lr = 1e-6
# batch_size = 32
# output_dim = 224, 224
# training_percentage = 0.8
# min_lr = 1e-12
# fixed_sigma = 1.667
# reduce_lr_factor = 0.2
# reduce_lr_patience = 5

if __name__ == '__main__':

    parser = create_parse()

    args = vars(parser.parse_args())

    dataset_path = args['dataset_path']

    metadata_path = args['metadata_path']

    output_folder = args['output_experiments']

    settings_path = args['settings']

    with open(settings_path) as json_file:
        params = json.load(json_file)

    epochs = params['epochs']

    batch_size = params['batch_size']

    output_dim = params['width'], params['height']

    unsharp_mask_multiplier = params['unsharp_mask_multiplier']

    initial_lr = params['initial_lr']

    model_name = params['model_name']

    min_lr = params['min_lr']

    unsharp_mask_filter = params['unsharp_mask_filter']

    fixed_sigma = params['fixed_sigma']

    augmentation_arguments = params['augmentation_params']

    reduce_lr_factor = params['reduce_lr_factor']

    reduce_lr_patience = params['reduce_lr_patience']

    early_stop_patience = params['early_stop_patience']

    val_split = params['val_split']

    test_split = params['test_split']

    now_dt_string = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")

    experiment_name = '{}-{}'.format(str(uuid.uuid1()), now_dt_string)

    print("Experiment name: %s" % experiment_name)

    experiment_output_folder = os.path.join(output_folder, experiment_name)

    os.makedirs(experiment_output_folder, exist_ok=True)

    model_output_path = os.path.join(experiment_output_folder, 'best-model')

    training_log_file = os.path.join(experiment_output_folder, 'training_log.csv')

    source_code = inspect.getsource(inspect.getmodule(inspect.currentframe()))

    source_code_path = os.path.join(experiment_output_folder, parser.prog)

    settings_output_path = os.path.join(experiment_output_folder, 'settings.json')

    arguments_output_path = os.path.join(experiment_output_folder, 'arguments.json')

    with open(arguments_output_path, 'w') as file:
        json.dump(args, file)

    with open(settings_output_path, 'w') as file:
        json.dump(params, file)

    with open(source_code_path, 'w') as file:
        file.write(source_code)

    with open(metadata_path) as json_file:
        dataset_metadata = json.load(json_file)

    generate_dataset = GenerateDataset(dataset_path, dataset_metadata)

    generate_dataset.load_dataset('None', val_split=val_split, test_split=test_split)

    dataset = generate_dataset.all_dataset()

    dataset_output_path = os.path.join(experiment_output_folder, 'dataset_files.json')

    with open(dataset_output_path, 'w') as file:
        json.dump(dataset, file)

    train_data = dataset['train']
    val_data = dataset['val']
    test_data = dataset['test']

    num_classes = generate_dataset.num_classes()

    print("Num classes: %s" % num_classes)

    image_generator = ImageDatasetGenerator(train_data['files'], train_data['labels'], batch_size, output_dim,
                                            dataset_metadata, train_mode=True, **augmentation_arguments)

    val_image_generator = ImageDatasetGenerator(val_data['files'], val_data['labels'], batch_size, output_dim,
                                                dataset_metadata, train_mode=False,
                                                random_crop=augmentation_arguments['random_crop'])

    test_image_generator = ImageDatasetGenerator(test_data['files'], test_data['labels'], batch_size, output_dim,
                                                 dataset_metadata, train_mode=False,
                                                 random_crop=augmentation_arguments['random_crop'])

    model_checkpoint = ModelCheckpoint(model_output_path, monitor='val_loss',
                                       save_best_only=True,
                                       save_weights_only=False)

    print_weights = LambdaCallback(
        on_epoch_end=lambda batch, logs: print_layer_weights(unsharp_mask_layer),
        on_epoch_begin=lambda batch, logs: print_layer_weights(unsharp_mask_layer))

    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=reduce_lr_factor, patience=reduce_lr_patience, min_lr=min_lr,
                          verbose=1),
        EarlyStopping(monitor='val_loss', patience=early_stop_patience, verbose=1),
        CSVLogger(training_log_file),
        model_checkpoint]

    input_layer = Input((output_dim[0], output_dim[1], 3))

    if unsharp_mask_filter == 'adaptiveLog':
        unsharp_mask_layer = UnsharpMaskLoG(kernel_size=(5, 5),
                                            regularizer_sigma=None,
                                            regularizer_amount=None,
                                            found_sigma=False,
                                            sigma=fixed_sigma)
        initial_layer = unsharp_mask_layer(input_layer)
        callbacks.append(print_weights)
    else:
        if unsharp_mask_filter == 'adaptive':
            unsharp_mask_layer = UnsharpMask(kernel_size=(5, 5),
                                             regularizer_sigma=None,
                                             regularizer_amount=None,
                                             found_sigma=False,
                                             sigma=fixed_sigma)
            initial_layer = unsharp_mask_layer(input_layer)
            callbacks.append(print_weights)
        else:
            initial_layer = input_layer

    output_layer = get_model((output_dim[0], output_dim[1], 3), initial_layer, model_name=model_name,
                             num_classes=num_classes,
                             trainable_layers_amount=0)

    final_model = Model(inputs=input_layer, outputs=output_layer)

    for layer in final_model.layers:
        print(layer)
        print(layer.name)

    final_model.summary()

    optimizer = LearningRateMultiplier(Adam(lr=initial_lr, decay=1e-4),
                                       lr_multipliers={'unsharp_mask': unsharp_mask_multiplier})

    final_model.compile(optimizer=optimizer,
                        loss='categorical_crossentropy', metrics=['accuracy', 'top_k_categorical_accuracy'])

    final_model.fit_generator(image_generator,
                              steps_per_epoch=calculate_steps(len(image_generator), batch_size),
                              validation_data=val_image_generator,
                              validation_steps=calculate_steps(len(val_image_generator), batch_size),
                              epochs=epochs,
                              verbose=1,
                              workers=100,
                              use_multiprocessing=True,
                              callbacks=callbacks)

    metrics_name = final_model.metrics_names

    evaluation = final_model.evaluate_generator(test_image_generator,
                                                steps=calculate_steps(len(test_image_generator), batch_size),
                                                verbose=1,
                                                workers=100,
                                                use_multiprocessing=True)

    metrics_dict = {metric_name: val for metric_name, val in zip(metrics_name, evaluation)}

    metrics_output_path = os.path.join(experiment_output_folder, 'metrics.json')

    with open(metrics_output_path, 'w') as json_file:
        json.dump(metrics_dict, json_file)

    predictions_test = final_model.predict_generator(test_image_generator,
                                                     steps=calculate_steps(len(test_image_generator), batch_size),
                                                     verbose=1,
                                                     workers=100,
                                                     use_multiprocessing=True
                                                     )
    y_predicted = predictions_test.argmax(axis=1)

    classes_names = generate_dataset.get_classes()

    columns_predictions = classes_names + ['groundtruth']

    y_true = np.array(test_data['labels'])

    # Saving predictions in a file.

    final_predictions = np.column_stack((predictions_test, y_true))

    df_predictions = pd.DataFrame(data=final_predictions, columns=columns_predictions)

    predictions_path = os.path.join(experiment_output_folder, 'predictions.tsv')

    df_predictions.to_csv(predictions_path, sep='\t', index=False)
