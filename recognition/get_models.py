# TODO! Log instead of print
import os
import zipfile
from os import path

import mxnet as mx


class get_models():
    """
    Download network-parameters from link
    """
    def __init__(self, all_messages, all_links, model_dir='models', dirname=None):
        """
        Network-parameters and some dataset pieces downloader initializer
        :param all_messages: messages that will printed out(only first 4 now)
        :param all_links: Download links
        :param model_dir: models download path
            DEFAULT='models'
        :param dirname: some dataset pieces. Fonts, typo
            DEFAULT=None
        """
        self.all_messages = all_messages
        self.all_links = all_links
        if dirname is None:
            self.dirname = 'dataset'
        self.model_dir = model_dir
        if not path.isdir(model_dir):
            os.makedirs(model_dir)

    def __call__(self):
        self.download_models()

    def __download(self, link):
        """
        Download link
        :param link: Download link
        :return:
        """
        mx.test_utils.download(link, dirname=self.model_dir)

    def __download_parameters(self, messages, link):
        """
        Network-parameter downloader initializer
        :param messages: messages that will printed out
        :param link: Download link
        :return:
        """
        print(messages)
        self.__download(link)

    def download_models(self):
        """
        Handles all downloading extracting process
        :return: Process finishes signature in bool. If process successful, returns True
        """
        model_dir = self.model_dir
        # %% Network-Parameters -> 0,1,2,3
        for idx in range(4):
            self.__download_parameters(self.all_messages[idx], self.all_links[idx])

        # %% Cost matrices -> 4,5,6,7
        print("Downloading cost matrices")
        for idx in range(4, 8):
            self.__download(self.all_links[idx])

        # %% Fonts -> 8
        print("Downloading fonts")
        dataset_dir = path.join('dataset', 'fonts')
        self.model_dir = dataset_dir
        if not path.isdir(dataset_dir):
            os.makedirs(dataset_dir)
        self.__download(self.all_links[8])
        with zipfile.ZipFile(path.join(dataset_dir, "fonts.zip"), "r") as zip_ref:
            zip_ref.extractall(dataset_dir)

        # %% Text datasets -> 9,10,11,12
        print("Downloading text datasets")
        dataset_dir = path.join('dataset', 'typo')
        if not path.isdir(dataset_dir):
            os.makedirs(dataset_dir)

        for idx in range(9, 13):
            self.__download(self.all_links[idx])

        self.model_dir = model_dir
        print("Finished")
        return True


if __name__ == "__main__":
    import time
    all_messages = [
        # Parameters
        "Downloading Paragraph Segmentation parameters",
        "Downloading Word Segmentation parameters",
        "Downloading Handwriting Line Recognition parameters",
        "Downloading Denoiser parameters",
        # Cost matrices
        "Downloading cost matrices",
        # Fonts
        "Downloading fonts",
        # Text datasets
        "Downloading text datasets"
    ]

    all_links = [
        # Parameters
        'https://s3.us-east-2.amazonaws.com/gluon-ocr/models/paragraph_segmentation2.params',
        'https://s3.us-east-2.amazonaws.com/gluon-ocr/models/word_segmentation2.params',
        'https://s3.us-east-2.amazonaws.com/gluon-ocr/models/handwriting_line8.params',
        'https://s3.us-east-2.amazonaws.com/gluon-ocr/models/denoiser2.params',
        # Cost matrices
        'https://s3.us-east-2.amazonaws.com/gluon-ocr/models/deletion_costs.txt',
        'https://s3.us-east-2.amazonaws.com/gluon-ocr/models/substitute_costs.txt',
        'https://s3.us-east-2.amazonaws.com/gluon-ocr/models/insertion_costs.txt',
        'https://s3.us-east-2.amazonaws.com/gluon-ocr/models/substitute_probs.json',
        # Fonts
        'https://s3.us-east-2.amazonaws.com/gluon-ocr/models/fonts.zip',
        # Text datasets
        'https://s3.us-east-2.amazonaws.com/gluon-ocr/models/alicewonder.txt',
        'https://s3.us-east-2.amazonaws.com/gluon-ocr/models/all.txt',
        'https://s3.us-east-2.amazonaws.com/gluon-ocr/models/text_train.txt',
        'https://s3.us-east-2.amazonaws.com/gluon-ocr/models/validating.json',
        'https://s3.us-east-2.amazonaws.com/gluon-ocr/models/typo-corpus-r1.txt'
    ]
    models = get_models(all_messages, all_links)
    t0 = time.time()
    models()
    print("Ellepsed time:", time.time() - t0)  # 47.976234436035156 second
