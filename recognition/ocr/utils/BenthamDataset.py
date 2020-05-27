# TODO! Complete Second dataset loader
import os

from .iam_dataset import IAMDataset


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
