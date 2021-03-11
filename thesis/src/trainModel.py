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
from layers.UnsharpMaskFixedLambda import UnsharpMaskFixedLambda
from callbacks.CustomModelCheckpoint import CustomModelCheckpoint
from models.Resnet34 import ResNet34
from optimizers.LearningRateMultiplier import LearningRateMultiplier
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.layers import Dense, Lambda, Input, GlobalAveragePooling2D, Dropout, Flatten, InputLayer
from tensorflow.keras.layers import Input, GaussianNoise, GaussianDropout
from tensorflow.keras import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, CSVLogger, \
    LambdaCallback
from tensorflow.keras.models import load_model
from skimage.filters import unsharp_mask
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2
import csv
import tensorflow as tf
from tensorflow.keras import backend as K

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# The GPU id to use, usually either "0" or "1"

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


FILTERS_TO_PRINT_SET = set(['adaptiveLog', 'adaptive'])


def write_weights(epoch, history_path, layer):
    with open(history_path, mode='a+', buffering=1) as history_amount_file:
        log_writer = csv.writer(history_amount_file, delimiter='\t',
                                quotechar='\'', quoting=csv.QUOTE_MINIMAL)
        log_writer.writerow([epoch, layer.get_weights()])


def print_layer_gradients(self, epoch, layer):
    loss = self.model.total_loss
    optimizer = self.model.optimizer
    gradients = optimizer.get_gradients(loss, layer.get_weights())
    print()
    print("Actual gradient of parameters in layer %s in epoch %s" % (layer.name, epoch))
    print(gradients)


def print_layer_weights(epoch, layer):
    print()
    print('Actual values of parameters in layer %s in epoch %s' % (layer.name, epoch))
    print(layer.get_weights())


def build_lenet_network(input_layer, num_classes):
    # Preprocess input with normalization.

    preprocessed_input = PreprocessImage('lenet')(input_layer)

    # convolutional block 1
    conv1 = layers.Conv2D(32, kernel_size=(5, 5), activation="relu", name="conv_1")(preprocessed_input)
    batch1 = layers.BatchNormalization(name="batch_norm_1")(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2), name="pool_1")(batch1)

    # convolutional block 2
    conv2 = layers.Conv2D(32, kernel_size=(10, 10), activation="relu", name="conv_2")(pool1)
    batch2 = layers.BatchNormalization(name="batch_norm_2")(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2), name="pool_2")(batch2)

    # convolutional block 3
    conv3 = layers.Conv2D(64, kernel_size=(15, 15), activation="relu", name="conv_3")(pool2)
    batch3 = layers.BatchNormalization(name="batch_norm_3")(conv3)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2), name="pool_3")(batch3)

    # fully connected layers
    # flatten = Flatten()(pool2)
    gap = layers.GlobalAveragePooling2D(name="gap")(pool3)
    d0 = layers.Dropout(rate=0.2, name="dropout0")(gap)
    fc1 = layers.Dense(256, activation="relu", name="fc1")(d0)
    d1 = layers.Dropout(rate=0.3, name="dropout1")(fc1)
    fc2 = layers.Dense(128, activation="relu", name="fc2")(d1)
    d2 = layers.Dropout(rate=0.4, name="dropout2")(fc2)

    # output layer
    output = layers.Dense(num_classes, activation="softmax")(d2)

    return output


def create_lenet(input_shape, input_layer, num_classes, dropout=0.5):
    x0 = PreprocessImage('lenet')(input_layer)

    x1 = layers.Conv2D(6, kernel_size=(5, 5), strides=(1, 1), activation='tanh', input_shape=input_shape,
                       padding="same")(x0)
    x2 = layers.AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding='valid')(x1)
    x3 = layers.Conv2D(16, kernel_size=(5, 5), strides=(1, 1), activation='tanh', padding='valid')(x2)
    x4 = layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x3)
    x5 = layers.Conv2D(120, kernel_size=(5, 5), strides=(1, 1), activation='tanh', padding='valid')(x4)
    x6 = layers.Flatten()(x5)
    x7 = layers.Dense(84, activation='tanh')(x6)
    output_layer = layers.Dense(num_classes, activation='softmax')(x7)

    return output_layer


def get_lambda_given_x(input_layer):
    # convolutional block 1
    conv1 = layers.Conv2D(32, kernel_size=(5, 5), activation="relu")(input_layer)
    batch1 = layers.BatchNormalization()(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(batch1)

    # fully connected layers
    gap1 = layers.GlobalAveragePooling2D()(pool1)
    dense1 = layers.Dense(256)(gap1)
    output = layers.Dense(1, activation='linear')(dense1)

    return output


def get_model(input_shape, input_layer, num_classes, model_name='resnet_50', trainable_layers_amount=0,
              dropout=0.5, path_model=None):
    if not path_model:
        if model_name == 'resnet_50':
            base_model = ResNet50(include_top=False,
                                  weights='imagenet',
                                  input_tensor=None,
                                  input_shape=input_shape)
        else:
            if model_name == 'mobile_net':
                base_model = MobileNet(include_top=False,
                                       weights='imagenet',
                                       input_tensor=None,
                                       input_shape=input_shape)
            elif model_name == 'resnet_34':
                base_model = ResNet34(include_top=False,
                                      weights='imagenet',
                                      input_tensor=None,
                                      input_shape=input_shape)
    else:
        print(f'Loading model in path {path_model}')
        loaded_model = load_model(path_model)
        base_model = loaded_model.layers[3]

    # Avoid training layers in resnet model.
    base_model.trainable = False

    layers = base_model.layers
    # Training the last
    if trainable_layers_amount != 0:
        if trainable_layers_amount != -1:
            trainable_layers = layers[-trainable_layers_amount:]
            assert len(trainable_layers) == trainable_layers_amount
        else:
            trainable_layers = layers
    else:
        trainable_layers = []

    for layer in trainable_layers:
        print("Making layer %s trainable " % layer.name)
        layer.trainable = True

    base_model.summary()
    x1 = PreprocessImage(model_name)(input_layer)
    x2 = base_model(x1, training=False)
    x3 = GlobalAveragePooling2D()(x2)
    x4 = Dense(1024, activation='relu')(x3)
    x5 = Dropout(dropout)(x4)
    output_layer = Dense(num_classes, activation='softmax', name='softmax', kernel_regularizer=l2(0.01),
                         bias_regularizer=l2(0.01), kernel_initializer=tf.keras.initializers.glorot_normal())(
        x5)
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

    parser.add_argument('--load_dataset', '-l',
                        type=bool,
                        required=False,
                        help='Flag to load an preexisted dataset.')

    parser.add_argument('--output_experiments', '-o',
                        required=True,
                        help='Folder of the experiments results')

    parser.add_argument('--settings', '-s',
                        required=True,
                        help='Setting of the experiment')

    parser.add_argument('--grid', '-g', required=True,
                        help='Name of the grid to run')

    parser.add_argument('--path_model', '-p', required=False,
                        help='Path to an existing model.')

    parser.add_argument('--fine_tune', '-f',
                        type=bool,
                        default=False,
                        required=False,
                        help='Flag to fine tune the model.')

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


def _on_epoch_begin(epoch, history_weights_path, unsharp_mask_layer):
    write_weights(epoch, history_weights_path, unsharp_mask_layer)
    print_layer_weights(epoch, unsharp_mask_layer)


def build_model(output_dim,
                num_classes,
                dropout,
                trainable_layers_amount,
                unsharp_mask_filter,
                kernel_size,
                fixed_sigma,
                model_name,
                callbacks,
                history_weights_path,
                path_model=None):
    input_layer = Input((output_dim[0], output_dim[1], 3))

    if unsharp_mask_filter == 'adaptiveLog':
        unsharp_mask_layer = UnsharpMaskLoG(kernel_size=(kernel_size, kernel_size),
                                            found_sigma=False,
                                            sigma=fixed_sigma)
        initial_layer = unsharp_mask_layer(input_layer)
    else:
        if unsharp_mask_filter == 'adaptive':
            unsharp_mask_layer = UnsharpMask(kernel_size=(kernel_size, kernel_size),
                                             found_sigma=False,
                                             sigma=fixed_sigma
                                             )
            initial_layer = unsharp_mask_layer(input_layer)
        else:
            if unsharp_mask_filter == 'adaptiveLambda':
                unsharp_mask_layer = UnsharpMaskFixedLambda(kernel_size=(kernel_size, kernel_size),
                                                            found_sigma=False,
                                                            sigma=fixed_sigma)

                lambda_generator = get_lambda_given_x(input_layer)

                initial_layer = unsharp_mask_layer([lambda_generator, input_layer])
            else:
                initial_layer = input_layer

    if unsharp_mask_filter in FILTERS_TO_PRINT_SET:
        with open(history_weights_path, mode='a+', buffering=1) as history_amount_file:
            log_writer = csv.writer(history_amount_file, delimiter='\t',
                                    quotechar='\'', quoting=csv.QUOTE_MINIMAL)
            log_writer.writerow(['epoch', 'amount'])

        print_weights = LambdaCallback(
            on_epoch_begin=lambda epoch, logs: _on_epoch_begin(epoch, history_weights_path, unsharp_mask_layer))
        callbacks.append(print_weights)

    if model_name != 'lenet':
        output_layer, base_model = get_model((output_dim[0], output_dim[1], 3), initial_layer,
                                             model_name=model_name,
                                             num_classes=num_classes,
                                             trainable_layers_amount=trainable_layers_amount,
                                             dropout=dropout,
                                             path_model=path_model)

        base_model.summary()
    else:
        output_layer = build_lenet_network(initial_layer, num_classes=num_classes)

    final_model = Model(inputs=input_layer, outputs=output_layer)

    return final_model, base_model


def run_experiment(args, params):
    dataset_path = args['dataset_path']

    metadata_path = args['metadata_path']

    output_folder = args['output_experiments']

    load_dataset = args['load_dataset']

    path_model = args['path_model']

    fine_tune = args['fine_tune']

    epochs = params['epochs']

    batch_size = params['batch_size']

    trainable_layers_amount = params['trainable_layers_amount']

    metric_stop = params['metric_stop']

    dropout = params['dropout']

    output_dim = params['width'], params['height']

    unsharp_mask_multiplier = params['unsharp_mask_multiplier']

    initial_lr = params['initial_lr']

    model_name = params['model_name']

    min_lr = params['min_lr']

    kernel_size = params['kernel_size']

    unsharp_mask_filter = params['unsharp_mask_filter']

    fixed_sigma = params['fixed_sigma']

    amount = params['amount']

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

    model_output_path = os.path.join(experiment_output_folder, 'best-model.ckpt')

    training_log_file = os.path.join(experiment_output_folder, 'training_log.csv')

    tensorboard_file = os.path.join(experiment_output_folder, 'tensorboard_log')

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

    if load_dataset:
        generate_dataset.load_dataset(dataset_path, val_split=val_split, test_split=test_split)
    else:
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

    preprocess_function = None

    if unsharp_mask_filter == 'unsharp_mask_scikit':
        preprocess_function = lambda x: unsharp_mask(x,
                                                     radius=fixed_sigma,
                                                     amount=amount,
                                                     multichannel=True,
                                                     preserve_range=True)

    image_generator = ImageDatasetGenerator(train_data['files'], train_data['labels'], batch_size, output_dim,
                                            dataset_metadata, train_mode=True, preprocess_function=preprocess_function,
                                            **augmentation_arguments)

    val_image_generator = ImageDatasetGenerator(val_data['files'], val_data['labels'], batch_size, output_dim,
                                                dataset_metadata, train_mode=False,
                                                random_crop=augmentation_arguments['random_crop'],
                                                preprocess_function=preprocess_function)

    test_image_generator = ImageDatasetGenerator(test_data['files'], test_data['labels'], batch_size, output_dim,
                                                 dataset_metadata, train_mode=False,
                                                 random_crop=augmentation_arguments['random_crop'],
                                                 preprocess_function=preprocess_function)

    print("Training data size: {}".format(len(image_generator)))
    print("Validation data size: {}".format(len(val_image_generator)))
    print("Test data size: {}".format(len(test_image_generator)))

    model_checkpoint = CustomModelCheckpoint(model_output_path, monitor='val_acc',
                                             save_best_only=True,
                                             save_weights_only=False)

    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir=tensorboard_file, histogram_freq=0),
        ReduceLROnPlateau(monitor=metric_stop, factor=reduce_lr_factor, patience=reduce_lr_patience, min_lr=min_lr,
                          verbose=1),
        EarlyStopping(monitor=metric_stop, patience=early_stop_patience, verbose=1),
        CSVLogger(training_log_file),
        model_checkpoint]

    final_model, base_model = build_model(output_dim, num_classes, dropout, trainable_layers_amount,
                                          unsharp_mask_filter, kernel_size,
                                          fixed_sigma, model_name, callbacks, history_weights_path, path_model)

    optimizer = Adam(lr=initial_lr, decay=1e-4)

    # optimizer = LearningRateMultiplier(op,
    #                                   lr_multipliers={'unsharp_mask': unsharp_mask_multiplier})

    final_model.compile(optimizer=optimizer,
                        loss='categorical_crossentropy', metrics=['accuracy', 'top_k_categorical_accuracy'])

    final_model.fit(image_generator,
                    steps_per_epoch=calculate_steps(len(image_generator), batch_size),
                    validation_data=val_image_generator,
                    validation_steps=calculate_steps(len(val_image_generator), batch_size),
                    epochs=epochs,
                    verbose=1,
                    workers=100,
                    use_multiprocessing=True,
                    callbacks=callbacks)

    if fine_tune:
        base_model.trainable = True
        for layer in base_model.layers[:100]:
            layer.trainable = False

        final_model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=initial_lr / 10),
                            loss='categorical_crossentropy', metrics=['accuracy', 'top_k_categorical_accuracy'])
        final_model.fit(image_generator,
                        steps_per_epoch=calculate_steps(len(image_generator), batch_size),
                        validation_data=val_image_generator,
                        validation_steps=calculate_steps(len(val_image_generator), batch_size),
                        epochs=20,
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

    save_predictions(val_data, val_image_generator, final_model, batch_size, classes_names,
                     predictions_path_val)

    predictions_path_test = os.path.join(experiment_output_folder, 'test_predictions.tsv')

    save_predictions(test_data, test_image_generator, final_model, batch_size, classes_names,
                     predictions_path_test)


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

    grid_name = args['grid']

    now_dt_string = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")

    grid_identifier = 'grid_%s' % now_dt_string

    output_grid = os.path.join(output, grid_identifier)

    os.mkdir(output_grid)

    args['output_experiments'] = output_grid

    print("Running grid: %s" % grid_identifier)

    epochs = 100
    dropout = 0.4
    lr = [1e-4]

    augmentation_params = {'vertical_flip': True, 'rotation': True, 'channel_shift_range': 10, 'shear_range': 0.2,
                           'horizontal_flip': True, 'random_crop': True}

    # models = ['mobile_net', 'resnet_34', 'resnet_50']
    models = []

    grid = {
        "batch_size": [64],
        "width": [224],
        "height": [224],
        "initial_lr": lr,
        "model_name": models,
        "min_lr": [1e-12],
        "epochs": [epochs],
        "fixed_sigma": [1.667],
        "reduce_lr_factor": [0.2],
        "reduce_lr_patience": [5],
        "early_stop_patience": [10],
        "kernel_size": [5],
        "val_split": [0.2],
        "test_split": [0.2],
        "trainable_layers_amount": [0],
        "augmentation_params": [augmentation_params],
        "metric_stop": ['val_accuracy'],
        "gpus": [1],
        "dropout": [dropout]
    }

    if grid_name == 'grid_no_filter' or grid_name == 'grid_transfer_bark':
        grid['unsharp_mask_filter'] = ['noFilter']
        grid['unsharp_mask_multiplier'] = [-1]
        grid['amount'] = [-1]
    else:
        if grid_name == 'grid_filter_scikit':
            grid['unsharp_mask_filter'] = ['unsharp_mask_scikit']
            grid['unsharp_mask_multiplier'] = [-1]
            grid['amount'] = [0.5, 1]
        else:
            grid['unsharp_mask_filter'] = ["adaptiveLog", "adaptive", "adaptiveLambda"]
            grid['unsharp_mask_multiplier'] = [100]
            grid['amount'] = [-1]

    configurations = create_configurations(grid)

    print("Total number of experiments: %s" % len(configurations))

    for num_exp, config in enumerate(configurations):
        print("Executing experiment number: %s" % num_exp)
        run_experiment(args, config)
        num_exp += 1

    print("Finishing experiments ...")
