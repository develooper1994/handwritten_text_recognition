import difflib
import importlib
import math
import random
import string

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
from PIL import Image
from skimage import transform as skimage_tf, exposure
import IPython.display
from IPython.display import Image
from tqdm import tqdm
import leven

import gluonnlp as nlp
import mxnet as mx
from mxnet.gluon.data.vision import datasets, transforms

from ocr.utils.expand_bounding_box import expand_bounding_box
from ocr.utils.sclite_helper import ScliteHelper
from ocr.utils.word_to_line import sort_bbs_line_by_line, crop_line_images
from ocr.utils.iam_dataset import IAMDataset, resize_image, crop_image, crop_handwriting_page
from ocr.utils.encoder_decoder import Denoiser, ALPHABET, encode_char, decode_char, EOS, BOS
from ocr.utils.beam_search import ctcBeamSearch

import ocr.utils.denoiser_utils
import ocr.utils.beam_search

from ocr.utils.denoiser_utils import SequenceGenerator

from ocr.paragraph_segmentation_dcnn import SegmentationNetwork, paragraph_segmentation_transform
from ocr.word_and_line_segmentation import SSD as WordSegmentationNet, predict_bounding_boxes
from ocr.handwriting_line_recognition import Network as HandwritingRecognitionNet, handwriting_recognition_transform
from ocr.handwriting_line_recognition import decode as decoder_handwriting, alphabet_encoding

from ocr.utils.preprocess import histogram, thresholds, CLAHE, filters, all_togather

ctx = mx.gpu(0) if mx.context.num_gpus() > 0 else mx.cpu()

random.seed(1)

form_size = (1120, 800)


class HTR:
    def __init__(self, image, form_size=(1120, 800), ctx=[mx.gpu(0)]):
        self.ctx = ctx
        self.form_size = form_size
        # loads with channels
        image = image[..., 0]  # converts gray scale
        # print(image.shape)
        image = image[np.newaxis, :]  # add batch dim
        # print(image.shape)
        self.image = mx.nd.array(image)  # converts to MXNet-NDarray

    def __call__(self, *args, **kwargs):
        pass

    def test(self):
        # original image
        image_name = "elyaz2.jpeg"
        image = mx.image.imread(image_name)  # 0 is grayscale
        self.image = image[..., 0]
        print(image.shape)

    def histogram(self, show=False):
        """
        Gives histogram, normalized cdf and bins values in numpy type
        :param show: Show histogram if show=True. default; show=False
        :return: histogram, normalized cdf and bins
        """
        hist, bins = np.histogram(self.image.flatten(), 256, [0, 256])
        cdf = hist.cumsum()
        cdf_normalized = cdf * hist.max() / cdf.max()
        if show:
            fig, ax = plt.subplots(1, figsize=(15, 9))
            plt.plot(cdf_normalized, color='b')
            plt.hist(self.image.flatten(), 256, [0, 256], color='r')
            plt.xlim([0, 256])
            plt.legend(('cdf', 'histogram'), loc='upper left')
            plt.show()
        return hist, cdf_normalized, bins

    def resize(self, show=False):
        # TODO: MXNET gpu -> cpu
        image = self.image.asnumpy()
        image_resize = cv2.resize(image, dsize=(form_size), interpolation=cv2.INTER_CUBIC)
        if show:
            fig, ax = plt.subplots(2, 1, figsize=(15, 18))
            print("test_image: ", image.shape)
            plt.subplot(121), plt.imshow(image.asnumpy(), cmap="gray")
            plt.subplot(121), plt.title("original image")
            plt.subplot(121), plt.axis("off")

            # downsampled image
            print("test_HTR_downsampled: ", image_resize.shape)
            plt.subplot(122), plt.imshow(image_resize, cmap="gray")
            plt.subplot(122), plt.title("downsampled image")
            plt.subplot(122), plt.axis("off")

            # save downsampled test_image and review
            plt.imsave("HTR_downsampled.jpeg", image_resize, cmap="gray")
