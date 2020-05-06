# TODO! Complete Second dataset loader
import os
import tarfile
import urllib
import sys
import time
import glob
import pickle
import xml.etree.ElementTree as ET
import cv2
import json
import numpy as np
import pandas as pd
import zipfile
import matplotlib.pyplot as plt
import logging

from mxnet.gluon.data import dataset
from mxnet import nd

from .expand_bounding_box import expand_bounding_box

from .iam_dataset import IAMDataset, crop_image, crop_handwriting_page, resize_image


class BenthamDataset(IAMDataset):
    MAX_IMAGE_SIZE_FORM = (1120, 800)
    MAX_IMAGE_SIZE_LINE = (60, 800)
    MAX_IMAGE_SIZE_WORD = (30, 140)

    def __init__(self, parse_method, credentials=None,
                 root=os.path.join(os.path.dirname(__file__), '..', '..', 'dataset', 'iamdataset'),
                 train=True, output_data="text",
                 output_parse_method=None,
                 output_form_text_as_array=False):
        super(BenthamDataset, self).__init__(parse_method, credentials=credentials,
                                             root=root,
                                             train=train, output_data=output_data,
                                             output_parse_method=output_parse_method,
                                             output_form_text_as_array=output_form_text_as_array)

        url = "http://www.transcriptorium.eu/~tsdata/BenthamR0/BenthamDatasetR0-Images.zip"
