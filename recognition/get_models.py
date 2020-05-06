import os
from os import path
import zipfile

import mxnet as mx

import threading
import asyncio
from multiprocessing.pool import ThreadPool

all_messages = [
    "Downloading Paragraph Segmentation parameters",
    "Downloading Word Segmentation parameters",
    "Downloading Handwriting Line Recognition parameters",
    "Downloading Denoiser parameters",
    "Downloading cost matrices",
    "Downloading fonts",
    "Downloading text datasets"
]

all_links = [
    'https://s3.us-east-2.amazonaws.com/gluon-ocr/models/paragraph_segmentation2.params',
    'https://s3.us-east-2.amazonaws.com/gluon-ocr/models/word_segmentation2.params',
    'https://s3.us-east-2.amazonaws.com/gluon-ocr/models/handwriting_line8.params',
    'https://s3.us-east-2.amazonaws.com/gluon-ocr/models/denoiser2.params',
    'https://s3.us-east-2.amazonaws.com/gluon-ocr/models/deletion_costs.txt',
    'https://s3.us-east-2.amazonaws.com/gluon-ocr/models/substitute_costs.txt',
    'https://s3.us-east-2.amazonaws.com/gluon-ocr/models/insertion_costs.txt',
    'https://s3.us-east-2.amazonaws.com/gluon-ocr/models/substitute_probs.json',
    'https://s3.us-east-2.amazonaws.com/gluon-ocr/models/fonts.zip',
    'https://s3.us-east-2.amazonaws.com/gluon-ocr/models/alicewonder.txt',
    'https://s3.us-east-2.amazonaws.com/gluon-ocr/models/all.txt',
    'https://s3.us-east-2.amazonaws.com/gluon-ocr/models/text_train.txt',
    'https://s3.us-east-2.amazonaws.com/gluon-ocr/models/validating.json',
    'https://s3.us-east-2.amazonaws.com/gluon-ocr/models/typo-corpus-r1.txt'
]


def download_parameters(messages, links, model_dir):
    print(messages)
    mx.test_utils.download(links, dirname=model_dir)


def download_models(dirname=None):
    if dirname is None:
        dirname = 'dataset'

    if not path.isdir(dirname):
        os.makedirs(dirname)

    model_dir = 'models'
    if not path.isdir(model_dir):
        os.makedirs(model_dir)

    #%% Parameters
    # print("Downloading Paragraph Segmentation parameters")
    # mx.test_utils.download('https://s3.us-east-2.amazonaws.com/gluon-ocr/models/paragraph_segmentation2.params',
    #                        dirname=model_dir)
    #
    # print("Downloading Word Segmentation parameters")
    # mx.test_utils.download('https://s3.us-east-2.amazonaws.com/gluon-ocr/models/word_segmentation2.params',
    #                        dirname=model_dir)
    #
    # print("Downloading Handwriting Line Recognition parameters")
    # mx.test_utils.download('https://s3.us-east-2.amazonaws.com/gluon-ocr/models/handwriting_line8.params',
    #                        dirname=model_dir)
    #
    # print("Downloading Denoiser parameters")
    # mx.test_utils.download('https://s3.us-east-2.amazonaws.com/gluon-ocr/models/denoiser2.params', dirname=model_dir)
    for idx in range(4):
        download_parameters(all_messages[idx], all_links[idx], model_dir)

    #%% Cost matrices
    print("Downloading cost matrices")
    mx.test_utils.download('https://s3.us-east-2.amazonaws.com/gluon-ocr/models/deletion_costs.txt', dirname=model_dir)
    mx.test_utils.download('https://s3.us-east-2.amazonaws.com/gluon-ocr/models/substitute_costs.txt',
                           dirname=model_dir)
    mx.test_utils.download('https://s3.us-east-2.amazonaws.com/gluon-ocr/models/insertion_costs.txt', dirname=model_dir)
    mx.test_utils.download('https://s3.us-east-2.amazonaws.com/gluon-ocr/models/substitute_probs.json',
                           dirname=model_dir)

    #%% Fonts
    print("Downloading fonts")
    model_dir = path.join('dataset', 'fonts')
    if not path.isdir(model_dir):
        os.makedirs(model_dir)
    mx.test_utils.download('https://s3.us-east-2.amazonaws.com/gluon-ocr/models/fonts.zip', dirname=model_dir)
    with zipfile.ZipFile(path.join(model_dir, "fonts.zip"), "r") as zip_ref:
        zip_ref.extractall(model_dir)

    #%% Text datasets
    print("Downloading text datasets")
    model_dir = path.join('dataset', 'typo')
    if not path.isdir(model_dir):
        os.makedirs(model_dir)

    mx.test_utils.download('https://s3.us-east-2.amazonaws.com/gluon-ocr/models/alicewonder.txt', dirname=model_dir)
    mx.test_utils.download('https://s3.us-east-2.amazonaws.com/gluon-ocr/models/all.txt', dirname=model_dir)
    mx.test_utils.download('https://s3.us-east-2.amazonaws.com/gluon-ocr/models/text_train.txt', dirname=model_dir)
    mx.test_utils.download('https://s3.us-east-2.amazonaws.com/gluon-ocr/models/validating.json', dirname=model_dir)
    mx.test_utils.download('https://s3.us-east-2.amazonaws.com/gluon-ocr/models/typo-corpus-r1.txt', dirname=model_dir)

    print("Finished")


if __name__ == "__main__":
    download_models()
