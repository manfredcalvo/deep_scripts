import os
import cv2
import pandas as pd
import numpy as np
import json
from distutils import util
import argparse
from collections import OrderedDict
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import log_loss
from sklearn.metrics import precision_recall_fscore_support
import torch
from PIL import Image
from torchvision import datasets, models, transforms
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras import layers
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras import Sequential, Model
from tensorflow.keras.optimizers import Adam
from skimage.filters import unsharp_mask
from tensorflow.keras.preprocessing import image
from dataset.generate_dataset import GenerateDataset
from dataset.ImageDatasetGenerator import ImageDatasetGenerator
from layers.PreprocessImage import PreprocessImage
from layers.UnsharpMaskLog import UnsharpMaskLoG
from layers.UnsharpMask import UnsharpMask
from layers.UnsharpMaskFixedLambda import UnsharpMaskFixedLambda
from optimizers.LearningRateMultiplier import LearningRateMultiplier
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model


# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def get_experiments_path(experiments_folder, grids_list):
    experiments_paths = []
    for grid in grids_list:
        grid_folder = os.path.join(experiments_folder, grid)
        grid_experiments = os.listdir(grid_folder)
        experiments_paths.extend(os.path.join(grid_folder, exp) for exp in grid_experiments)
    return experiments_paths


def get_probs_no_crops(n_images, probs_crops, idx_rows):
    probs_no_crops = []
    for i in range(n_images):
        probs_row = probs_crops[idx_rows[i]:idx_rows[i + 1]].mean(axis=0)
        probs_no_crops.append(probs_row)
    probs_no_crops = np.array(probs_no_crops)
    return probs_no_crops


def get_crops_and_labels(dataset, target_size=(224, 224), resize_dim=None, pytorch=False):
    out_height, out_width = target_size
    crops = []
    files = dataset['files']
    labels = dataset['labels']
    crops_labels = []
    idx_rows = [0]
    for path, label in zip(files, labels):
        img = cv2.imread(path)
        if resize_dim:
            img = cv2.resize(img, resize_dim)
        if pytorch:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        (height, width) = img.shape[0], img.shape[1]
        crops_height = height // out_height
        crops_width = width // out_width
        # print("Processing image in path: {}".format(path))
        idx_rows.append(idx_rows[-1] + crops_height * crops_width)
        for i in range(crops_height):
            for j in range(crops_width):
                crops.append(img[i * out_height:(i + 1) * out_height, j * out_width:(j + 1) * out_width, :])
                crops_labels.append(label)
    return np.array(crops), np.array(crops_labels), np.array(idx_rows)


def predict_crops(crops, model, pytorch=False):
    if pytorch:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        with torch.no_grad():
            print("Predicting using pytorch model")

            data_transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            print("Normalizing images before prediction")

            images_transformed = torch.stack([data_transforms(Image.fromarray(crop)) for crop in crops])
            predictions_test = []
            batch_size = 64
            index = 0
            print('Predicting images')

            for inputs in images_transformed[index:index + batch_size]:
                inputs = inputs.to(device)
                logits = model(inputs)
                probs = torch.nn.functional.softmax(logits, dim=1)
                batch_preds = probs.detach().cpu().numpy()
                predictions_test.append(batch_preds)
                index = index + batch_size
            return np.concatenate(predictions_test)
    else:
        return model.predict(crops)


def get_accuracy_crops(model, dataset, split='val', resize_dim=None, pytorch=False):
    val_dataset = dataset[split]
    crops, crops_labels, idx_rows = get_crops_and_labels(val_dataset, resize_dim=resize_dim, pytorch=pytorch)
    probs_crops = predict_crops(crops, model, pytorch)
    probs_no_crops = get_probs_no_crops(len(val_dataset['labels']), probs_crops, idx_rows)
    preds_no_crops = probs_no_crops.argmax(axis=1)
    preds_crops = probs_crops.argmax(axis=1)
    acc_no_crops = accuracy_score(np.array(val_dataset['labels']), preds_no_crops)
    acc_crops = accuracy_score(crops_labels, preds_crops)
    top_acc_no_crops = top_k_categorical(np.array(val_dataset['labels']), probs_no_crops)
    top_acc_crops = top_k_categorical(crops_labels, probs_crops)
    return acc_crops, top_acc_crops, acc_no_crops, top_acc_no_crops


def top_k_categorical(y_true, y_probs, k=5):
    predicted_labels = np.argsort(-1 * y_probs, axis=1)[:, :k]
    hits = 0
    n = y_true.shape[0]
    for i in range(n):
        if y_true[i] in predicted_labels[i]:
            hits += 1
    return hits / n


def get_confusion_matrix(y_true, y_pred, species_name):
    cm = confusion_matrix(y_true, y_pred)
    cm = cm / cm.sum(axis=1)[:, None]
    cm_pd = pd.DataFrame(cm, columns=species_name, index=species_name)
    return cm_pd


def get_predictions(predictions_df):
    probs = predictions_df.values[:, :-1]
    y_true = predictions_df.values[:, -1]
    y_pred = probs.argmax(axis=1)
    return y_true, y_pred, probs


def load_dataset(experiment_path):
    dataset_path = os.path.join(experiment_path, 'dataset_files.json')
    with open(dataset_path, 'r') as file:
        dataset = json.load(file)
    return dataset


def load_experiment_model(experiment_path, pytorch=False):
    model_path = os.path.join(experiment_path, 'best-model.ckpt')
    if pytorch:
        # Loading pytorch model.
        model = torch.load(model_path)
        model.eval()
    else:
        # Loading keras model.
        model = load_model(model_path)
        model.summary()
    return model


def get_metrics(y_true, y_pred, species_name):
    precision, recall, fscore, support = precision_recall_fscore_support(y_true, y_pred, average=None)
    data = {'precision': precision, 'recall': recall, 'fscore': fscore, 'count': support, 'especie': species_name}
    df_metrics = pd.DataFrame(data)[['especie', 'recall', 'count', 'fscore', 'precision']]
    df_metrics = df_metrics.set_index('especie')
    return df_metrics


def process_experiment(experiment_path, resize_dim=None, pytorch=False):
    evaluation = OrderedDict()
    predictions_path = os.path.join(experiment_path, 'val_predictions.tsv')
    test_predictions_path = os.path.join(experiment_path, 'test_predictions.tsv')
    if not os.path.exists(predictions_path):
        return
    predictions = pd.read_csv(predictions_path, sep='\t')
    predictions_test = pd.read_csv(test_predictions_path, sep='\t')
    species_name = predictions.columns[:-1]
    y_true, y_pred, y_probs = get_predictions(predictions)
    y_true_t, y_pred_t, y_probs_t = get_predictions(predictions_test)
    accuracy_test = accuracy_score(y_true_t, y_pred_t)
    accuracy = accuracy_score(y_true, y_pred)
    loss = log_loss(y_true, y_probs)
    cm = get_confusion_matrix(y_true, y_pred, species_name)
    cm_path = os.path.join(experiment_path, 'val_confusion_matrix.csv')
    cm.to_csv(cm_path, sep='\t', index=False)
    df_metrics = get_metrics(y_true, y_pred, species_name)
    metrics_path = os.path.join(experiment_path, 'val_metrics.csv')
    df_metrics.to_csv(metrics_path, sep='\t', index=False)
    model = load_experiment_model(experiment_path, pytorch)
    dataset = load_dataset(experiment_path)
    acc_crops, top_k_crops, acc_multi_crops, top_k_multi_crops = get_accuracy_crops(model, dataset, split='val',
                                                                                    resize_dim=resize_dim,
                                                                                    pytorch=pytorch)
    acc_crops_test, top_k_crops_test, acc_multi_crops_test, top_k_multi_crops_test = get_accuracy_crops(model, dataset,
                                                                                                        split='test',
                                                                                                        resize_dim=resize_dim,
                                                                                                        pytorch=pytorch)
    evaluation['accuracy_crops_test'] = acc_crops_test
    evaluation['top_k_categorical_crops_test'] = top_k_crops_test
    evaluation['accuracy_multi_crops_test'] = acc_multi_crops_test
    evaluation['top_k_categorical_multi_crops_test'] = top_k_multi_crops_test
    evaluation['accuracy_crops'] = acc_crops
    evaluation['top_k_categorical_crops'] = top_k_crops
    evaluation['accuracy_multi_crops'] = acc_multi_crops
    evaluation['top_k_categorical_multi_crops'] = top_k_multi_crops
    evaluation['accuracy'] = accuracy
    evaluation['accuracy_test'] = accuracy_test
    evaluation['top_k_categorical'] = top_k_categorical(y_true, y_probs)
    evaluation['logloss'] = loss
    for metric in df_metrics.columns:
        evaluation[metric] = df_metrics[metric].to_dict()
    evaluation['val_confusion_matrix'] = cm_path
    evaluation['val_metrics'] = metrics_path
    evaluation['species'] = list(species_name)
    evaluation['experiment_path'] = experiment_path
    with open(os.path.join(experiment_path, 'settings.json'), 'r') as file:
        parameters = json.load(file)
    for param, val in OrderedDict(parameters).items():
        evaluation[param] = val
    return evaluation


def process_experiments(experiments_paths, resize_dim=None, pytorch=False):
    evaluations = []
    total_exps = len(experiments_paths)
    print("Total experiments to process: {}".format(total_exps))
    for n_experiment, experiment_path in enumerate(experiments_paths):
        print("Processing experiment: {} path: {}".format(n_experiment, experiment_path))
        evaluation = process_experiment(experiment_path, resize_dim, pytorch)
        if evaluation:
            evaluations.append(evaluation)
    return pd.DataFrame(evaluations)


def get_result_by_model(df_results, model_name):
    df_results_model = df_results[df_results['model_name'] == model_name]
    return df_results_model.sort_values(by='accuracy', ascending=False)


def create_parse():
    parser = argparse.ArgumentParser(
        description='Training deep neuronal network to predict the specie of an specimen represented by an image'
    )

    parser.add_argument('--base_experiments_path', '-be',
                        required=True,
                        help='Path of the experiments')

    parser.add_argument('--grids_list', '-gl',
                        nargs='+',
                        type=str,
                        required=True,
                        help='List of grids')

    parser.add_argument('--output_path', '-o',
                        required=True,
                        help='Path of the output csv')

    parser.add_argument('--pytorch_experiments',
                        default=False,
                        type=lambda x: bool(util.strtobool(x)),
                        help='Whether the experiments were run on pytorch.')

    return parser


if __name__ == '__main__':
    parser = create_parse()

    args = vars(parser.parse_args())

    base_experiments_path = args['base_experiments_path']

    grids_list = args['grids_list']

    output_path = args['output_path']

    pytorch_experiments = args['pytorch_experiments']

    print(args)

    experiments_paths = get_experiments_path(base_experiments_path, grids_list)

    df_results = process_experiments(experiments_paths, pytorch=pytorch_experiments)

    df_results.to_csv(output_path)
