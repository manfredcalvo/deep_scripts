import os
import json
import argparse
import uuid
from datetime import datetime
import inspect
from itertools import product
import pandas as pd
import math
import numpy as np
from dataset.generate_dataset import GenerateDataset
from dataset.ImageDatasetGenerator import ImageDatasetGenerator
from layers.PreprocessImage import PreprocessImage
from layers.UnsharpMaskLog import UnsharpMaskLoG
from layers.UnsharpMask import UnsharpMask
from optimizers.LearningRateMultiplier import LearningRateMultiplier
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.layers import Dense, Lambda, Input, GlobalAveragePooling2D, Dropout, Flatten
from tensorflow.keras.layers import Input, GaussianNoise, GaussianDropout
from tensorflow.keras import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, CSVLogger, ModelCheckpoint, \
    LambdaCallback
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras.models import load_model
from tensorflow.keras.regularizers import l2
from tensorflow import keras
import csv
import tensorflow as tf


# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# The GPU id to use, usually either "0" or "1"

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def write_weights(epoch, history_path, layer):
    with open(history_path, mode='a+', buffering=1) as history_amount_file:
        log_writer = csv.writer(history_amount_file, delimiter='\t',
                                quotechar='\'', quoting=csv.QUOTE_MINIMAL)
        log_writer.writerow([epoch, layer.get_weights()])


def print_layer_weights(epoch, layer):
    print()
    print('Actual values of parameters in layer %s in epoch %s' % (layer.name, epoch))
    print(layer.get_weights())


def create_lenet(input_shape, input_layer, num_classes, dropout=0.5):
    x1 = tf.keras.layers.Conv2D(filters=6, kernel_size=(3, 3), activation='relu', input_shape=input_shape)(input_layer)

    x2 = tf.keras.layers.AveragePooling2D()(x1)

    x3 = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu')(x2)

    x4 = tf.keras.layers.AveragePooling2D()(x3)

    x5 = tf.keras.layers.Flatten()(x4)

    x6 = tf.keras.layers.Dense(units=120, activation='relu')(x5)

    x7 = tf.keras.layers.Dense(units=84, activation='relu')(x6)

    output_layer = tf.keras.layers.Dense(units=num_classes, activation='softmax')(x7)

    return output_layer


def get_model(input_shape, input_layer, num_classes, model_name='resnet_50', trainable_layers_amount=0, architecture=1,
              dropout=0.5):
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
        else:
            if model_name == 'inception_v3':
                base_model = InceptionV3(include_top=False,
                                         weights='imagenet',
                                         input_tensor=None,
                                         input_shape=input_shape)

    layers = base_model.layers

    # Avoid training layers in resnet model.
    for layer in layers:
        layer.trainable = False

    # Training the last
    if trainable_layers_amount != 0:
        trainable_layers = layers[-trainable_layers_amount:]
        assert len(trainable_layers) == trainable_layers_amount
    else:
        trainable_layers = []

    for layer in trainable_layers:
        print("Making layer %s trainable " % layer.name)
        layer.trainable = True

    x1 = PreprocessImage(model_name)(input_layer)
    x2 = base_model(x1)

    if architecture == 1:
        # Top layer 1
        x3 = Flatten()(x2)
        x4 = Dropout(dropout)(x3)
    else:
        x3 = GlobalAveragePooling2D()(x2)
        x4 = Dropout(dropout)(x3)

    output_layer = Dense(num_classes, activation='softmax', name='softmax', kernel_regularizer=l2(0.01),
                         bias_regularizer=l2(0.01))(x4)

    return output_layer, base_model


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



def save_predictions(dataset, dataset_generator, model, batch_size, classes_names, predictions_path):
    predictions_test = model.predict_generator(dataset_generator,
                                               steps=calculate_steps(len(dataset_generator), batch_size),
                                               verbose=1,
                                               workers=100,
                                               use_multiprocessing=True
                                               )

    columns_predictions = classes_names + ['groundtruth']

    y_true = np.array(dataset['labels'])

    # Saving predictions in a file.

    final_predictions = np.column_stack((predictions_test, y_true))

    df_predictions = pd.DataFrame(data=final_predictions, columns=columns_predictions)

    df_predictions.to_csv(predictions_path, sep='\t', index=False)


def run_experiment(args, params):
    dataset_path = args['dataset_path']

    metadata_path = args['metadata_path']

    output_folder = args['output_experiments']

    epochs = params['epochs']

    batch_size = params['batch_size']

    trainable_layers_amount = params['trainable_layers_amount']

    dropout = params['dropout']

    architecture = params['architecture']

    output_dim = params['width'], params['height']

    unsharp_mask_multiplier = params['unsharp_mask_multiplier']

    initial_lr = params['initial_lr']

    model_name = params['model_name']

    min_lr = params['min_lr']

    kernel_size = params['kernel_size']

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
    print("Params: %s" % str(params))

    experiment_output_folder = os.path.join(output_folder, experiment_name)

    os.makedirs(experiment_output_folder, exist_ok=True)

    model_output_path = os.path.join(experiment_output_folder, 'best-model')

    training_log_file = os.path.join(experiment_output_folder, 'training_log.csv')

    source_code = inspect.getsource(inspect.getmodule(inspect.currentframe()))

    source_code_path = os.path.join(experiment_output_folder, parser.prog)

    settings_output_path = os.path.join(experiment_output_folder, 'settings.json')

    arguments_output_path = os.path.join(experiment_output_folder, 'arguments.json')

    history_weights_path = os.path.join(experiment_output_folder, 'history_weights.csv')

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

    model_checkpoint = ModelCheckpoint(model_output_path, monitor='val_acc',
                                       save_best_only=True,
                                       save_weights_only=False)

    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=reduce_lr_factor, patience=reduce_lr_patience, min_lr=min_lr,
                          verbose=1),
        EarlyStopping(monitor='val_loss', patience=early_stop_patience, verbose=1),
        CSVLogger(training_log_file),
        model_checkpoint]

    input_layer = Input((output_dim[0], output_dim[1], 3))

    # TODO: Probar un size diferente del filtro. Mas grande y mas pequenno.
    # TODO: Revisar tammano imagenes en el paper de Jose Carranza.
    # TODO: Revisar resultado de los filtros en las imagenes luego de entrenar el filtro.
    # TODO: Usar VGG16.

    with open(history_weights_path, mode='a+', buffering=1) as history_amount_file:

        log_writer = csv.writer(history_amount_file, delimiter='\t',
                                quotechar='\'', quoting=csv.QUOTE_MINIMAL)
        log_writer.writerow(['epoch', 'amount'])

    print_weights = LambdaCallback(
        on_epoch_end=lambda epoch, logs: write_weights(epoch, history_weights_path, unsharp_mask_layer),
        on_epoch_begin=lambda epoch, logs: print_layer_weights(epoch, unsharp_mask_layer))

    if unsharp_mask_filter == 'adaptiveLog':
        unsharp_mask_layer = UnsharpMaskLoG(kernel_size=(kernel_size, kernel_size),
                                            regularizer_sigma=None,
                                            regularizer_amount=None,
                                            found_sigma=False,
                                            sigma=fixed_sigma)
        initial_layer = unsharp_mask_layer(input_layer)
        callbacks.append(print_weights)
    else:
        if unsharp_mask_filter == 'adaptive':
            unsharp_mask_layer = UnsharpMask(kernel_size=(kernel_size, kernel_size),
                                             found_sigma=False,
                                             sigma=fixed_sigma)
            initial_layer = unsharp_mask_layer(input_layer)
            callbacks.append(print_weights)
        else:
            initial_layer = input_layer

    if architecture in [1, 2]:
        output_layer, base_model = get_model((output_dim[0], output_dim[1], 3), initial_layer, model_name=model_name,
                                             num_classes=num_classes,
                                             trainable_layers_amount=trainable_layers_amount,
                                             architecture=architecture,
                                             dropout=dropout)

        base_model.summary()
    else:
        output_layer = create_lenet((output_dim[0], output_dim[1], 3), initial_layer, num_classes=num_classes,
                                    dropout=dropout)

    gpus = 2

    if gpus <= 1:
        final_model = Model(inputs=input_layer, outputs=output_layer)
    else:
        with tf.device("/cpu:0"):
            # initialize the model
            final_model = Model(inputs=input_layer, outputs=output_layer)

            # make the model parallel
            final_model = multi_gpu_model(final_model, gpus=gpus)

    for layer in final_model.layers:
        print(layer)
        print(layer.name)

    final_model.summary()

    layer_name = initial_layer.name

    optimizer = LearningRateMultiplier(Adam(lr=initial_lr, decay=1e-4),
                                       lr_multipliers={layer_name: unsharp_mask_multiplier})

    '''
    optimizer = LearningRateMultiplier(SGD(lr=initial_lr, momentum=0.99),
                                       lr_multipliers={layer_name: unsharp_mask_multiplier})
    '''

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

    final_model = load_model(model_output_path)

    evaluation = final_model.evaluate_generator(test_image_generator,
                                                steps=calculate_steps(len(test_image_generator), batch_size),
                                                verbose=1,
                                                workers=100,
                                                use_multiprocessing=True)

    metrics_dict = {metric_name: val for metric_name, val in zip(metrics_name, evaluation)}

    metrics_output_path = os.path.join(experiment_output_folder, 'metrics.json')

    with open(metrics_output_path, 'w') as json_file:
        json.dump(metrics_dict, json_file)

    classes_names = generate_dataset.get_classes()

    predictions_path_val = os.path.join(experiment_output_folder, 'val_predictions.tsv')

    save_predictions(val_data, val_image_generator, final_model, batch_size, classes_names, predictions_path_val)

    predictions_path_test = os.path.join(experiment_output_folder, 'test_predictions.tsv')

    save_predictions(test_data, test_image_generator, final_model, batch_size, classes_names, predictions_path_test)


def create_configurations(dic_input):
    configurations = []
    for values in product(*dic_input.values()):
        dict_result = dict(zip(dic_input.keys(), values))
        configurations.append(dict_result)
    return configurations


if __name__ == '__main__':

    parser = create_parse()

    args = vars(parser.parse_args())

    output = args['output_experiments']

    now_dt_string = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")

    grid_identifier = 'grid_%s' % now_dt_string

    output_grid = os.path.join(output, grid_identifier)

    os.mkdir(output_grid)

    args['output_experiments'] = output_grid

    print("Running grid: %s" % grid_identifier)

    epochs = 200
    dropout = 0.4
    lr = [1e-4]

    grid_no_filter = {
        "batch_size": [64],
        "width": [224],
        "height": [224],
        "initial_lr": lr,
        "model_name": ["resnet_50"],
        "min_lr": [1e-12],
        "epochs": [epochs],
        "unsharp_mask_filter": ["noFilter"],
        "fixed_sigma": [1.667],
        "reduce_lr_factor": [0.1],
        "reduce_lr_patience": [3],
        "early_stop_patience": [5],
        "kernel_size": [5],
        "val_split": [0.2],
        "test_split": [0.2],
        "trainable_layers_amount": [0, 10, 20],
        "unsharp_mask_multiplier": [1],
        "augmentation_params": [{"random_crop": True,
                                 # "rotation_range": 90, No
                                 # "width_shift_range": 0.2,
                                 # "height_shift_range": 0.2,
                                 "shear_range": 0.2,
                                 # "zoom_range": 0.1,
                                 "channel_shift_range": 10,
                                 "horizontal_flip": True,
                                 "vertical_flip": True,
                                 # "fill_mode": 'nearest'
                                 }],
        "architecture": [2],
        "dropout": [dropout]
    }

    grid_filter = {
        "batch_size": [64],
        "width": [224],
        "height": [224],
        "initial_lr": lr,
        "model_name": ["resnet_50"],
        "min_lr": [1e-12],
        "epochs": [epochs],
        "unsharp_mask_filter": ["adaptive", "adaptiveLog"],
        "fixed_sigma": [1.667],
        "trainable_layers_amount": [10, 20],
        "reduce_lr_factor": [0.1],
        "reduce_lr_patience": [3],
        "early_stop_patience": [5],
        "kernel_size": [10, 5],
        "val_split": [0.2],
        "test_split": [0.2],
        "unsharp_mask_multiplier": [1e4],
        "augmentation_params": [{"random_crop": True,
                                 # "rotation_range": 90, No
                                 # "width_shift_range": 0.2,
                                 # "height_shift_range": 0.2,
                                 "shear_range": 0.2,
                                 # "zoom_range": 0.1,
                                 "channel_shift_range": 10,
                                 "horizontal_flip": True,
                                 "vertical_flip": True,
                                 # "fill_mode": 'nearest'
                                 }],
        "architecture": [2],
        "dropout": [dropout]
    }

    configurations_no_filter = create_configurations(grid_no_filter)
    configurations_filter = create_configurations(grid_filter)

    configurations = configurations_no_filter + configurations_filter

    print("Total number of experiments: %s" % len(configurations))

    for num_exp, config in enumerate(configurations):
        print("Executing experiment number: %s" % num_exp)
        run_experiment(args, config)
        num_exp += 1

    print("Finishing experiments ...")
