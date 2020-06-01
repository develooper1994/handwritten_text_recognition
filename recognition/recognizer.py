## ideas:
#     - TODO! Change all numeric types to mxnet to gain more speed.
#     - TODO! Add error handling mechanism.
#     - TODO! Add visualization module to handle inspection
#     - TODO! Add training classes to handle in one-step all
#     - TODO! Split data(mxnet split) to faster training and ?inference?
#     - TODO! Logging instead of printing
#     - weighted levenshtein
#     - re-trained the language model on GBW [~ didn't work too well]
#     - only penalize non-existing words
#     - Add single word training for denoiser
#     - having 2 best edit distance rather than single one
#     - split sentences based on punctuation
#     - use CTC loss for ranking
#     - meta model to learn to weight the scores from each thing

# %% Standart Python modules
import asyncio
import logging
import logging.config
import random

# %% helper modules
from pprint import pprint

# %% mxnet modules
import gluonnlp as nlp

# %% numerical modules
import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from skimage import exposure
from tqdm import tqdm

# %% my modules
from detection.craft_text_detector.craft_text_detector.imgproc import read_image

try:
    from recognition.get_models import async_get_models as get_models
    from recognition.ocr.handwriting_line_recognition import Network as HandwritingRecognitionNet, \
        handwriting_recognition_transform
    from recognition.ocr.paragraph_segmentation_dcnn import SegmentationNetwork, paragraph_segmentation_transform
    from recognition.ocr.utils.denoiser_utils import SequenceGenerator
    from recognition.ocr.utils.encoder_decoder import Denoiser, ALPHABET, encode_char, EOS, BOS
    from recognition.ocr.utils.expand_bounding_box import expand_bounding_box
    from recognition.ocr.utils.iam_dataset import crop_handwriting_page
    from recognition.ocr.utils.sclite_helper import ScliteHelper
    from recognition.ocr.utils.word_to_line import sort_bbs_line_by_line, crop_line_images
    from recognition.ocr.word_and_line_segmentation import SSD as WordSegmentationNet, predict_bounding_boxes
    from recognition.utils.recognizer_utils import *
except:
    try:
        from handwritten_text_recognition.recognition.get_models import async_get_models as get_models
        from handwritten_text_recognition.recognition.ocr.handwriting_line_recognition import \
            Network as HandwritingRecognitionNet, handwriting_recognition_transform
        from handwritten_text_recognition.recognition.ocr.paragraph_segmentation_dcnn import \
            SegmentationNetwork, paragraph_segmentation_transform
        from handwritten_text_recognition.recognition.ocr.utils.denoiser_utils import SequenceGenerator
        from handwritten_text_recognition.recognition.ocr.utils.encoder_decoder import Denoiser, ALPHABET, \
            encode_char, EOS, BOS
        from handwritten_text_recognition.recognition.ocr.utils.expand_bounding_box import expand_bounding_box
        from handwritten_text_recognition.recognition.ocr.utils.iam_dataset import crop_handwriting_page
        from handwritten_text_recognition.recognition.ocr.utils.sclite_helper import ScliteHelper
        from handwritten_text_recognition.recognition.ocr.utils.word_to_line import sort_bbs_line_by_line, \
            crop_line_images
        from handwritten_text_recognition.recognition.ocr.word_and_line_segmentation import \
            SSD as WordSegmentationNet, predict_bounding_boxes
        from handwritten_text_recognition.recognition.utils.recognizer_utils import *
    except:
        from recognition.handwritten_text_recognition.recognition.get_models import async_get_models as get_models
        from recognition.handwritten_text_recognition.recognition.ocr.handwriting_line_recognition import \
            Network as HandwritingRecognitionNet, \
            handwriting_recognition_transform
        from recognition.handwritten_text_recognition.recognition.ocr.paragraph_segmentation_dcnn import \
            SegmentationNetwork, paragraph_segmentation_transform
        from recognition.handwritten_text_recognition.recognition.ocr.utils.denoiser_utils import SequenceGenerator
        from recognition.handwritten_text_recognition.recognition.ocr.utils.encoder_decoder import Denoiser, ALPHABET, \
            encode_char, EOS, BOS
        from recognition.handwritten_text_recognition.recognition.ocr.utils.expand_bounding_box import \
            expand_bounding_box
        from recognition.handwritten_text_recognition.recognition.ocr.utils.iam_dataset import crop_handwriting_page
        from recognition.handwritten_text_recognition.recognition.ocr.utils.sclite_helper import ScliteHelper
        from recognition.handwritten_text_recognition.recognition.ocr.utils.word_to_line import sort_bbs_line_by_line, \
            crop_line_images
        from recognition.handwritten_text_recognition.recognition.ocr.word_and_line_segmentation import \
            SSD as WordSegmentationNet, predict_bounding_boxes
        from recognition.handwritten_text_recognition.recognition.utils.recognizer_utils import *

random.seed(1)

logging.basicConfig(filename="recognizer.py.log",
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    filemode='w')
logger = logging.getLogger("root")
# Setting the threshold of logger to DEBUG
logger.setLevel(logging.DEBUG)


# # Test messages
# logger.debug("Harmless debug Message")
# logger.info("Just an information")
# logger.warning("Its a Warning")
# logger.error("Did you try to divide by zero")
# logger.critical("Internet is down")

# CRITICAL = 50
# FATAL = CRITICAL
# ERROR = 40
# WARNING = 30
# WARN = WARNING
# INFO = 20
# DEBUG = 10
# NOTSET = 0


def log_print(message, level=logging.DEBUG, show=False):
    logger.setLevel(level)
    if show:
        print(message, __name__)
    else:
        message = "{} {}".format(message, str(__name__))
        logger.log(level, message)


# exception_logger.py
@singleton
class Logger:
    def __init__(self, level=logging.DEBUG):
        """
        Creates a logging object and returns it
        """
        self.logger = logging.getLogger("root")
        self.fh = logging.FileHandler(r"recognizer.py.log")
        self.logger.setLevel(level)
        # create the logging file handler
        fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        formatter = logging.Formatter(fmt)
        self.fh.setFormatter(formatter)
        # add handler to logger object
        self.logger.addHandler(self.fh)

    def __call__(self):
        return self.logger

    def exception(self):
        """
        A decorator that wraps the passed in function and logs
        exceptions should one occur

        @param logger: The logging object
        """

        def decorator(func):
            def wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except:
                    # log the exception
                    err = "There was an exception in  "
                    err += func.__name__
                    self.logger.exception(err)
                # re-raise the exception
                raise

            return wrapper

        return decorator

    # TODO! !!! Not ready !!!
    def log_print(self, message, show="print"):
        def decorator(func):
            message = "{} {}".format(message, str(func.__name__))

            def wrapper(*args, **kwargs):
                if show == "print":
                    print(message)
                elif show == "logging":
                    logger.log(self.level, message)
                elif show == "both":
                    print(message)
                    logger.log(self.level, message)
                elif show == "somewhere":
                    return func(*args, **kwargs)
                else:
                    assert True, "I don't know what can i do for you"

            return wrapper

        return decorator


# logger = Logger()

# form_size = (1120, 800)

# Algorithm:
# 1) Paragraph ->
# 2) words ->
# 3) word to line(to protect the information context) ->
# 4) word image to string line by line
class recognize:
    """
    The main, One-step module is it.
    Usage example:
        image = mx.image.imread("tests/TurkishHandwritten/elyaz2.jpeg")
        image = image.asnumpy()
        recog = recognize(image, device=device)
        result = recog()
    """

    def __init__(self, image, net_parameter_path=None, form_size=(1120, 800), device=None, crop=False,
                 ScliteHelperPATH=None, show=False, is_test=False):
        """
        Handwritten Text Recognization in one step
        :param image: input image in numpy.array object that includes handwritten text
        :param net_parameter_pathname: network(model) parameter paths with name
        :param form_size: possible form size
            DEFAULT=(1120, 800)
        :param device: determines the device that the model(or network) will work on
            DEFAULT=None
        If it is None:
            device = mx.cpu()
        :param crop: cropping detected text area
            DEFAULT=False
        :param ScliteHelperPATH: Tool that helps to get quantitative results. https://github.com/usnistgov/SCTK
            DEFAULT=None
        :param show: Show plot if show=True. default; show=False
            DEFAULT=False
        :param is_test: If it is True than activate SCTK tool to get quantative results.
            DEFAULT=False
        """
        self.reload(ScliteHelperPATH, crop, device, form_size, image, is_test, net_parameter_path, show)

    def reload(self, ScliteHelperPATH, crop, device, form_size, image, is_test, net_parameter_path, show):
        # %% Default-Parameters
        self.__set_default_parameters(image, form_size, device, crop, ScliteHelperPATH, show, is_test)
        # %% Network-Parameters
        self.__set_default_networks(net_parameter_path)

    def set_image(self, image):
        """
        Configure input image. if image is string then tries to access path.
        :param image: input image or input image path
        :return: input image
        """
        self.image = image  # consider image is numpy-array or some tensor
        if isinstance(image, str):
            # consider image is path of image
            self.image = read_image(image)  # numpy image
        return self.image

    def reload_default_parameters(self, image, form_size, device, crop, ScliteHelperPATH, show, is_test):
        self.set_image(image)
        assert type(self.image) is np.ndarray, "Please enter numpy array"
        # self.image = mx.nd.array(image)  # converts to MXNet-NDarray
        # self.image = self.image.asnumpy()
        # loads with channels
        if len(self.image.shape) > 2:
            self.image = self.image[..., 0]  # converts gray scale
        # print(image.shape)
        # self.image = self.image[np.newaxis, :]  # add batch dim
        # # print(image.shape)
        self.form_size = form_size
        if device is None:
            device = mx.cpu()
        self.device = device
        self.crop = crop
        if ScliteHelperPATH is None:
            ScliteHelperPATH = '../SCTK/bin'
        if is_test:
            self.sclite = ScliteHelper(ScliteHelperPATH)
        self.is_test = is_test
        self.show = show
        # network hyperparameters
        self.reload_network_hyperparameters()
        # download_models()

        self.gray_scale=True

    def reload_network_hyperparameters(self, predicted_text_area=0, croped_image=0, predicted_bb=0,
                                       min_c=0.1, overlap_thres=0.1, topk=600,
                                       segmented_paragraph_size=None, line_image_size=None,
                                       line_images_array=None, character_probs=None):
        self.predicted_text_area = predicted_text_area
        self.croped_image = croped_image
        self.predicted_bb = predicted_bb  # TODO! change for external detection.
        self.min_c = min_c
        self.overlap_thres = overlap_thres
        self.topk = topk

        self.segmented_paragraph_size = segmented_paragraph_size
        if segmented_paragraph_size is None:
            self.segmented_paragraph_size = (700, 700)

        self.line_image_size = line_image_size
        if line_image_size is None:
            self.line_image_size = (60, 800)

        self.line_images_array = line_images_array
        if line_images_array is None:
            self.line_images_array = []

        self.character_probs = character_probs
        if character_probs is None:
            self.character_probs = []

    def __set_default_parameters(self, image, form_size, device, crop, ScliteHelperPATH, show, is_test):
        self.reload_default_parameters(image, form_size, device, crop, ScliteHelperPATH, show, is_test)

    def __set_default_networks(self, net_parameter_path):
        # !!! slower while loding async this function !!!
        # run it async
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.__load_all_networks(net_parameter_path))

        ## We use a language model in order to rank the propositions from the denoiser
        language_model, vocab = nlp.model.big_rnn_lm_2048_512(dataset_name='gbw', pretrained=True,
                                                              ctx=self.device)
        moses_tokenizer = nlp.data.SacreMosesTokenizer()
        moses_detokenizer = nlp.data.SacreMosesDetokenizer()

        # !!! Slowest loading!
        # We use beam search to sample the output of the denoiser
        log_print("Beam sampler created")
        beam_sampler = nlp.model.BeamSearchSampler(beam_size=20,
                                                   decoder=self.denoiser.decode_logprob,
                                                   eos_id=EOS,
                                                   scorer=nlp.model.BeamSearchScorer(),
                                                   max_length=150)
        log_print("Sequence generator created")
        self.generator = SequenceGenerator(beam_sampler, language_model, vocab, self.device,
                                           moses_tokenizer,
                                           moses_detokenizer)

    async def __load_all_networks(self, net_parameter_path):
        net_parameter_paths = self.load_parameter_paths(net_parameter_path)
        self.net_parameter_paths = net_parameter_paths
        self.denoiser_net_parameter_path = self.net_parameter_paths[0]
        self.handwriting_line_recognition_net_parameter_path = self.net_parameter_paths[1]
        self.paragraph_segmentation_net_parameter_path = self.net_parameter_paths[2]
        self.word_segmentation_net_parameter_path = self.net_parameter_paths[3]

        # load networks async. more stable time variation
        await asyncio.wait([
            self.__load_HandwritingRecognitionNet(),
            self.__load_DenoiserNet(),
            self.__load_WordSegmentationNet(),
            self.__load_SegmentationNetwork()
        ])

    async def __load_HandwritingRecognitionNet(self):
        await asyncio.sleep(0.01)  # event loop runs function for a while suspends
        log_print("Handwriting line segmentation model loading")
        self.handwriting_line_recognition_net = HandwritingRecognitionNet(rnn_hidden_states=512,
                                                                          rnn_layers=2, ctx=self.device,
                                                                          max_seq_len=160)
        self.handwriting_line_recognition_net.load_parameters(self.handwriting_line_recognition_net_parameter_path,
                                                              ctx=self.device)
        self.handwriting_line_recognition_net.hybridize()
        log_print("Handwriting line segmentation model loading completed")

    async def __load_DenoiserNet(self):
        await asyncio.sleep(0.01)  # event loop runs function for a while suspends
        # We use a seq2seq denoiser to translate noisy input to better output
        log_print("Denoiser model loading")
        self.FEATURE_LEN = 150
        self.denoiser = Denoiser(alphabet_size=len(ALPHABET), max_src_length=self.FEATURE_LEN,
                                 max_tgt_length=self.FEATURE_LEN,
                                 num_heads=16, embed_size=256, num_layers=2)
        self.denoiser.load_parameters(self.denoiser_net_parameter_path, ctx=self.device)
        self.denoiser.hybridize(static_alloc=True)
        log_print("Denoiser model loading completed")

    async def __load_WordSegmentationNet(self):
        await asyncio.sleep(0.01)  # event loop runs function for a while suspends
        log_print("Word segmentation model loading")
        self.word_segmentation_net = WordSegmentationNet(2, ctx=self.device)
        self.word_segmentation_net.load_parameters(self.word_segmentation_net_parameter_path)
        self.word_segmentation_net.hybridize()
        log_print("Word segmentation model loading completed")

    async def __load_SegmentationNetwork(self):
        await asyncio.sleep(0.01)  # event loop runs function for a while suspends
        log_print("Paragraph segmentation model loading")
        self.paragraph_segmentation_net = SegmentationNetwork(ctx=self.device)
        self.paragraph_segmentation_net.cnn.load_parameters(self.paragraph_segmentation_net_parameter_path,
                                                            ctx=self.device)
        self.paragraph_segmentation_net.hybridize()
        log_print("Paragraph segmentation model loading completed")

    def load_parameter_paths(self, net_parameter_path=None):
        # assert not isinstance(net_parameter_pathname, list) or isinstance(net_parameter_pathname, tuple), \
        #     "Please enter net_parameter_pathname in List or tuple type"

        if net_parameter_path is None:
            # models must be sorted by ascending order!
            net_parameter_path = "models/"
        net_parameter_pathname = [
            net_parameter_path+"denoiser2.params",
            net_parameter_path+"handwriting_line8.params",
            net_parameter_path+"paragraph_segmentation2.params",
            net_parameter_path+"word_segmentation2.params",
        ]
        if net_parameter_path == "download".lower():
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
            models()
        return net_parameter_pathname

    def __call__(self, *args, **kwargs):
        return self.one_step(*args, **kwargs)

    def one_step(self, expand_bb_scale_x=0.18, expand_bb_scale_y=0.23, segmented_paragraph_size=(700, 700)):
        """
        Calculate all in of them in one (long) step
        :param expand_bb_scale_x: Scale constant along x axis
            DEFAULT=0.18
        :param expand_bb_scale_y: Scale constant along y axis
            DEFAULT=0.23
        :param segmented_paragraph_size: segmented paragraph size in tuple
            DEFAULT=(700, 700)
        :return: all calculated results.
            results = {
                'predicted_text_area': predicted_text_area,
                'croped_image': croped_image,
                'predicted_bb': predicted_bb,
                'line_images_array': line_images_array,
                'character_probs': character_probs,
                'decoded': decoded
            }
        """
        # detection
        predicted_text_area, croped_image, predicted_bb = self.make_detection(expand_bb_scale_x, expand_bb_scale_y,
                                                                              segmented_paragraph_size)
        # recognition
        line_images_array, character_probs, decoded = self.make_recognition()
        # decoded_line_ams, decoded_line_bss, decoded_line_denoisers = decoded
        all_results = {
            'predicted_text_area': predicted_text_area,
            'croped_image': croped_image,
            'predicted_bb': predicted_bb,
            'line_images_array': line_images_array,
            'character_probs': character_probs,
            'decoded': decoded
        }
        return all_results

    def make_detection(self, expand_bb_scale_x, expand_bb_scale_y, segmented_paragraph_size):
        """
        Making detection
        :param expand_bb_scale_x: scale constant for x axis
        :param expand_bb_scale_y: scale constant for y axis
        :param segmented_paragraph_size: segmented paragraph size
        :return: predicted_text_area, croped_image, predicted_bb
        """
        croped_image, predicted_text_area = self.image_preprocess(expand_bb_scale_x, expand_bb_scale_y,
                                                                  segmented_paragraph_size)
        predicted_bb = self.word_detection()
        return predicted_text_area, croped_image, predicted_bb

    def make_recognition(self):
        """
        Making recognition
        :return: line_images_array, character_probs, [decoded_line_ams, decoded_line_bss, decoded_line_denoisers]
        """
        line_images_array = self.word_to_line()
        character_probs = self.handwriting_recognition_probs(line_images_array=line_images_array)
        decoded = self.qualitative_result()
        # decoded_line_ams, decoded_line_bss, decoded_line_denoisers = decoded
        return line_images_array, character_probs, decoded

    def image_preprocess(self, expand_bb_scale_x, expand_bb_scale_y, segmented_paragraph_size):
        predicted_text_area = self.predict_bbs(expand_bb_scale_x=expand_bb_scale_x, expand_bb_scale_y=expand_bb_scale_y)
        croped_image = None
        if self.crop:
            croped_image = self.crop_image(segmented_paragraph_size)
        return croped_image, predicted_text_area

    # %% network functions
    ## Paragraph segmentation
    # Given the image of a form in the IAM dataset, predict a bounding box of the handwriten component. The model was trained on using https://github.com/ThomasDelteil/Gluon_OCR_LSTM_CTC/blob/master/paragraph_segmentation_dcnn.py and an example is presented in https://github.com/ThomasDelteil/Gluon_OCR_LSTM_CTC/blob/master/paragraph_segmentation_dcnn.ipynb
    def predict_bbs(self, expand_bb_scale_x=0.18, expand_bb_scale_y=0.23):
        """
        Predicts bounding box of given image to detect where the text in the image
        :param expand_bb_scale_x: Scale constant along x axis
            DEFAULT=0.18
        :param expand_bb_scale_y: Scale constant along y axis
            DEFAULT=0.23
        :return: predicted bounding box for hole text area
        """
        resized_image = paragraph_segmentation_transform(self.image, self.form_size)
        # print(resized_image.shape)
        bb_predicted = self.paragraph_segmentation_net(resized_image.as_in_context(self.device))
        bb_predicted = bb_predicted[0].asnumpy()
        # all train set was in the middle
        self.predicted_text_area = expand_bounding_box(bb_predicted,
                                                       expand_bb_scale_x=expand_bb_scale_x,
                                                       expand_bb_scale_y=expand_bb_scale_y)
        if self.show:
            # s_y, s_x = int(i/2), int(i%2)
            _, ax = plt.subplots(1, figsize=(15, 18))
            ax.imshow(self.image, cmap='Greys_r')
            (x, y, w, h) = bb_predicted
            image_h, image_w = self.image.shape[-2:]
            (x, y, w, h) = (x * image_w, y * image_h, w * image_w, h * image_h)
            rect = patches.Rectangle((x, y), w, h, fill=False, color="r", ls="--")
            ax.add_patch(rect)
            ax.axis('off')
        return self.predicted_text_area

    ## Image Processing
    # Crop the handwriting component out of the original IAM form.
    def crop_image(self, segmented_paragraph_size=(700, 700)):
        """
        Crops predicted bounding box.
        :param segmented_paragraph_size: segmented paragraph size in tuple
            DEFAULT=(700, 700)
        :return: croped image
        """
        # segmented_paragraph_size = (700, 700)

        bb = self.predicted_text_area
        croped_image = crop_handwriting_page(self.image, bb, image_size=segmented_paragraph_size)
        self.croped_image = croped_image

        # from IPython.display import Image
        # Image(image)
        if self.show:
            _, _ = plt.subplots(1, figsize=(15, 18))  # Just determines figure size
            plt.imshow(croped_image, cmap='Greys_r')
            plt.draw()
            plt.axis('off')

        return croped_image

    ## Line/word segmentation
    # Given a form with only handwritten text, predict a bounding box for each word.The model was trained with https://github.com / ThomasDelteil / Gluon_OCR_LSTM_CTC / blob / language_model / word_segmentation.py
    def word_detection(self, image=None):
        """
        Word detector with SSD(single shot detection)
        :return: predicted bounding box for each word
        """
        # paragraph_segmented_image = paragraph_segmented_images[0]
        if image is None:
            image = self.image
        paragraph_segmented_image = image
        self.predicted_bb = predict_bounding_boxes(self.word_segmentation_net, paragraph_segmented_image, self.min_c,
                                                   self.overlap_thres,
                                                   self.topk, self.device)

        if self.show:
            fig, ax = plt.subplots(1, figsize=(15, 10))
            ax.imshow(paragraph_segmented_image, cmap='Greys_r')
            for j in range(self.predicted_bb.shape[0]):
                (x, y, w, h) = self.predicted_bb[j]
                image_h, image_w = paragraph_segmented_image.shape[-2:]
                (x, y, w, h) = (x * image_w, y * image_h, w * image_w, h * image_h)

                rect = patches.Rectangle((x, y), w, h, fill=False, color="r")
                ax.add_patch(rect)
                ax.axis('off')

            plt.show()
            plt.draw()
            fig.savefig("test_word_segmentation.jpg")

        return self.predicted_bb

    ## Word to line image processing
    # Algorithm to sort then group all words within a line together.
    def word_to_line(self, image=None):
        """
        Converts word bounding boxes to line bounding boxes by overlapping.
        :param image: Image in numpy format. There is if you want to change image after
        :return: croped line images array.
        """
        if image is None:
            image = self.image
        paragraph_segmented_image = image

        line_bbs = sort_bbs_line_by_line(self.predicted_bb, y_overlap=0.4)
        line_images = crop_line_images(paragraph_segmented_image, line_bbs)
        self.line_images_array.append(line_images)

        if self.show:
            fig, ax = plt.subplots(figsize=(15, 18))

            ax.imshow(paragraph_segmented_image, cmap='Greys_r')
            ax.axis('off')
            for line_bb in line_bbs:
                (x, y, w, h) = line_bb
                image_h, image_w = paragraph_segmented_image.shape[-2:]
                (x, y, w, h) = (x * image_w, y * image_h, w * image_w, h * image_h)

                rect = patches.Rectangle((x, y), w, h, fill=False, color="r")
                ax.add_patch(rect)

            plt.show()
            plt.draw()
            fig.savefig("Word _to_line.jpg")
        return self.line_images_array

    ## Handwriting recognition
    # Given each line of text, predict a string of the handwritten text. This network was trained with https://github.com/ThomasDelteil/Gluon_OCR_LSTM_CTC/blob/language_model/handwriting_line_recognition.py
    def handwriting_recognition_probs(self, line_images_array=None):
        """
        Calculates character probabilities
        :return: Character probabilities
        """
        if line_images_array is None:
            line_images_array = self.line_images_array

        for line_images in line_images_array:
            form_character_prob = []
            for i, line_image in enumerate(line_images):
                line_image = handwriting_recognition_transform(line_image, self.line_image_size)
                line_character_prob = self.handwriting_line_recognition_net(line_image.as_in_context(self.device))
                form_character_prob.append(line_character_prob)
            self.character_probs.append(form_character_prob)
        return self.character_probs

    ## Denoising the text output
    def get_denoised(self, prob, ctc_bs=False):
        """
        Returns denoised encoder
        :param prob: Probabilities
        :param ctc_bs: Contextual switch
            DEFAULT=False
        :return: Denoised encoder
        """
        if ctc_bs:  # Using ctc beam search before denoising yields only limited improvements a is very slow
            text = get_beam_search(prob)
        else:
            text = get_arg_max(prob)
        src_seq, src_valid_length = encode_char(text)
        src_seq = mx.nd.array([src_seq], ctx=self.device)
        src_valid_length = mx.nd.array(src_valid_length, ctx=self.device)
        encoder_outputs, _ = self.denoiser.encode(src_seq, valid_length=src_valid_length)
        states = self.denoiser.decoder.init_state_from_encoder(encoder_outputs,
                                                               encoder_valid_length=src_valid_length)
        inputs = mx.nd.full(shape=(1,), ctx=src_seq.context, dtype=np.float32, val=BOS)
        # TODO! mxnet.base.MXNetError:
        # [23:54:01] c:\jenkins\workspace\mxnet-tag\mxnet\src\storage\./pooled_storage_manager.h:161:
        # cudaMalloc retry failed: out of memory
        output = self.generator.generate_sequences(inputs, states, text)
        return output.strip()

    # %% getting results
    ## Qualitative Result
    # !!! Most important function that gives final results. !!!
    def qualitative_result(self):
        """
        - [AM] Arg Max CTC Decoding
        - [BS] Beam Search CTC Decoding
        - [D ] Adding Text Denoiser
        :return: [AM], [BS], [D]
        """
        decoded_line_ams = []
        decoded_line_bss = []
        decoded_line_denoisers = []
        # really shitty solution but it worked.
        if not self.show:
            for i, form_character_probs in enumerate(self.character_probs):
                for j, line_character_probs in enumerate(form_character_probs):
                    decoded_line_am = get_arg_max(line_character_probs)
                    log_print("[AM] %s" % str(decoded_line_am))
                    decoded_line_ams.append(decoded_line_am)
                    decoded_line_bs = get_beam_search(line_character_probs)
                    decoded_line_bss.append(decoded_line_bs)
                    decoded_line_denoiser = self.get_denoised(line_character_probs.asnumpy(), ctc_bs=False)
                    log_print("[D] %s" % str(decoded_line_denoiser))
                    decoded_line_denoisers.append(decoded_line_denoiser)
        else:
            for i, form_character_probs in enumerate(self.character_probs):
                fig, axs = plt.subplots(len(form_character_probs) + 1,
                                        figsize=(10, int(1 + 2.3 * len(form_character_probs))))
                for j, line_character_probs in enumerate(form_character_probs):
                    decoded_line_am = get_arg_max(line_character_probs)
                    log_print("[AM] %s" % str(decoded_line_am))
                    decoded_line_ams.append(decoded_line_am)
                    decoded_line_bs = get_beam_search(line_character_probs)
                    decoded_line_bss.append(decoded_line_bs)
                    decoded_line_denoiser = self.get_denoised(line_character_probs.asnumpy(), ctc_bs=False)
                    log_print("[D] %s" % str(decoded_line_denoiser))
                    decoded_line_denoisers.append(decoded_line_denoiser)

                    line_image = self.line_images_array[i][j]
                    axs[j].imshow(line_image.squeeze(), cmap='Greys_r')
                    axs[j].set_title(
                        "[AM]: {}\n[BS]: {}\n[D ]: {}\n\n".format(decoded_line_am, decoded_line_bs,
                                                                  decoded_line_denoiser),
                        fontdict={"horizontalalignment": "left", "family": "monospace"}, x=0)
                    axs[j].axis('off')
                axs[-1].imshow(np.zeros(shape=self.line_image_size), cmap='Greys_r')
                axs[-1].axis('off')
        return decoded_line_ams, decoded_line_bss, decoded_line_denoisers

    ## Quantitative Results
    # Iterative through the test data with the previous tests to obtain the total Character Error Rate (CER).

    # %%
    # TODO! NOT TESTED YET!!!
    # only unix-like and linux
    #
    # git clone https://github.com/usnistgov/SCTK
    # cd SCTK
    # export CXXFLAGS="-std=c++11" && make config
    # make all
    # make check
    # make install
    # make doc
    # cd -

    def get_qualitative_results_lines(self, denoise_func):
        """
        Get all quantative "character error results" for each line
        :param denoise_func: denoiser function
        :return: CER values
        """
        self.sclite.clear()
        test_ds_line = IAMDataset("line", train=False)
        for i in tqdm(range(1, len(test_ds_line))):
            image, text = test_ds_line[i]
            line_image = exposure.adjust_gamma(image, 1)
            line_image = handwriting_recognition_transform(line_image, self.line_image_size)
            character_probabilities = self.handwriting_line_recognition_net(line_image.as_in_context(self.device))
            decoded_text = denoise_func(character_probabilities)
            actual_text = text[0].replace("&quot;", '"').replace("&apos;", "'").replace("&amp;", "&")
            self.sclite.add_text([decoded_text], [actual_text])

        cer, er = self.sclite.get_cer()
        log_print("Mean CER = {}".format(cer))
        return cer

    def get_qualitative_results(self, denoise_func, credentials=None):
        """
        Get all quantative "character error results" full pipeline
        :param credentials: credentials to access and download IAMdataset
        :param denoise_func: denoiser function
            DEFAULT=None
        :return: CER values
        """

        error_message = """
        Please enter creditentials in json string format to access, download and preprocess IAMdataset
        For example;
        {
          "username": "<USERNAME>",
          "password": "<PASSWORD>"
        }"""
        assert not credentials is None, error_message

        self.sclite.clear()
        test_ds = get_IAMDataset_test()
        for i in tqdm(range(1, len(test_ds))):
            image, text = test_ds[i]
            resized_image = paragraph_segmentation_transform(image, image_size=self.form_size)
            paragraph_bb = self.paragraph_segmentation_net(resized_image.as_in_context(self.device))
            paragraph_bb = paragraph_bb[0].asnumpy()
            paragraph_bb = expand_bounding_box(paragraph_bb, expand_bb_scale_x=0.01,
                                               expand_bb_scale_y=0.01)
            paragraph_segmented_image = crop_handwriting_page(image, paragraph_bb,
                                                              image_size=self.segmented_paragraph_size)
            word_bb = predict_bounding_boxes(self.word_segmentation_net, paragraph_segmented_image, self.min_c,
                                             self.overlap_thres, self.topk,
                                             self.device)
            line_bbs = sort_bbs_line_by_line(word_bb, y_overlap=0.4)
            line_images = crop_line_images(paragraph_segmented_image, line_bbs)

            predicted_text = []
            for line_image in line_images:
                line_image = exposure.adjust_gamma(line_image, 1)
                line_image = handwriting_recognition_transform(line_image, self.line_image_size)
                character_probabilities = self.handwriting_line_recognition_net(line_image.as_in_context(self.device))
                decoded_text = denoise_func(character_probabilities)
                predicted_text.append(decoded_text)

            actual_text = text[0].replace("&quot;", '"').replace("&apos;", "'").replace("&amp;", "&")
            actual_text = actual_text.split("\n")
            if len(predicted_text) > len(actual_text):
                predicted_text = predicted_text[:len(actual_text)]
            self.sclite.add_text(predicted_text, actual_text)

        cer, _ = self.sclite.get_cer()
        print("Mean CER = {}".format(cer))
        logger.info("Mean CER = {}".format(cer))
        return cer

    def get_quantative_all(self):
        """
        Get all quantative "character error results"
        :return: CER values
        """
        # %% md
        # CER with pre - segmented lines
        CER = []
        CER0 = self.get_qualitative_results_lines(get_arg_max)
        log_print("CER0 = {}".format(CER0))
        CER.append(CER0)
        CER1 = self.get_qualitative_results_lines(self.get_denoised)
        log_print("CER1 = {}".format(CER1))
        CER.append(CER1)

        # %% md
        # CER full pipeline
        CER2 = self.get_qualitative_results(get_arg_max)
        log_print("CER2 = {}".format(CER2))
        CER.append(CER2)

        CER3 = self.get_qualitative_results(get_beam_search)
        log_print("CER3 = {}".format(CER3))
        CER.append(CER3)

        # %%
        cer_denoiser = self.get_qualitative_results(self.get_denoised)
        log_print("cer_denoiser = {}".format(cer_denoiser))
        CER.append(cer_denoiser)

        return CER

    ## dummy test
    def tasteit(self):
        """
        Just a dummy test
        :return: None
        """
        sentence = "This sentnce has an eror"
        src_seq, src_valid_length = encode_char(sentence)
        src_seq = mx.nd.array([src_seq], ctx=self.device)
        src_valid_length = mx.nd.array(src_valid_length, ctx=self.device)
        encoder_outputs, _ = self.denoiser.encode(src_seq, valid_length=src_valid_length)
        states = self.denoiser.decoder.init_state_from_encoder(encoder_outputs,
                                                               encoder_valid_length=src_valid_length)
        inputs = mx.nd.full(shape=(1,), ctx=src_seq.context, dtype=np.float32, val=BOS)
        log_print("sentence = {}".format(sentence))
        log_print("Choise")

        generated = self.generator.generate_sequences(inputs, states, sentence)
        log_print("generated = {}".format(generated))


if __name__ == "__main__":
    num_device = 1
    device_queue = "cpu"
    device = device_selection_helper(device=device_queue, num_device=num_device, framework="mxnet")

    # %% recognize class
    image_name = "a1.png"
    image_path = r"C:\Users\selcu\PycharmProjects\ocr_toolkit\detection\craft_text_detector\figures\IAM8" + "/" + image_name
    # image_name = "elyaz2.jpeg"
    # image_path = r"tests/TurkishHandwritten" + "/" + image_name
    image = mx.image.imread(image_path)
    image = image.asnumpy()
    # import time
    # t0 = time.time()
    recog = recognize(image, device=device)

    # async -> 12.5510573387146 second
    # sync -> 12.63401460647583
    # print("Ellepsed time:", time.time()-t0)

    result = recog()

    pprint(result)
