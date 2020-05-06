## ideas:
#     - TODO! Change all numeric types to mxnet to gain more speed.
#     - TODO! Add error handling mechanism.
#     - TODO! Add visualization module to handle inspection
#     - TODO! Add training classes to handle in one-step all
#     - TODO! Split data to faster training and ?inference?
#     - weighted levenshtein
#     - re-trained the language model on GBW [~ didn't work too well]
#     - only penalize non-existing words
#     - Add single word training for denoiser
#     - having 2 best edit distance rather than single one
#     - split sentences based on punctuation
#     - use CTC loss for ranking
#     - meta model to learn to weight the scores from each thing
from pprint import pprint
import random
import time
from os import listdir
from os.path import isfile, join

import cv2
import gluonnlp as nlp
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import mxnet as mx
import numpy as np
from skimage import exposure
from tqdm import tqdm

from recognition.get_models import download_models

from recognition.ocr.handwriting_line_recognition import Network as HandwritingRecognitionNet, \
    handwriting_recognition_transform
from recognition.ocr.handwriting_line_recognition import decode as decoder_handwriting, alphabet_encoding
from recognition.ocr.paragraph_segmentation_dcnn import SegmentationNetwork, paragraph_segmentation_transform
from recognition.ocr.utils.beam_search import ctcBeamSearch
from recognition.ocr.utils.denoiser_utils import SequenceGenerator
from recognition.ocr.utils.encoder_decoder import Denoiser, ALPHABET, encode_char, EOS, BOS
from recognition.ocr.utils.expand_bounding_box import expand_bounding_box
from recognition.ocr.utils.iam_dataset import IAMDataset, crop_handwriting_page
from recognition.ocr.utils.preprocess import histogram, all_togather
from recognition.ocr.utils.sclite_helper import ScliteHelper
from recognition.ocr.utils.word_to_line import sort_bbs_line_by_line, crop_line_images
from recognition.ocr.word_and_line_segmentation import SSD as WordSegmentationNet, predict_bounding_boxes

random.seed(1)


# form_size = (1120, 800)

# Algorithm:
# 1) Paragraph ->
# 2) words ->
# 3) word to line(to protect the information context) ->
# 4) word image to string line by line

## Character Probalities to Text
def get_arg_max(prob):
    """
    The greedy algorithm convert the output of the handwriting recognition network into strings.
    :param prob: probability values
    :return: maximum probabilities
    """
    arg_max = mx.nd.array(prob).topk(axis=2).asnumpy()
    return decoder_handwriting(arg_max)[0]


def get_beam_search(prob, width=5):
    possibilities = ctcBeamSearch(prob.softmax()[0].asnumpy(), alphabet_encoding, None, width)
    return possibilities[0]


## recognizer Class to handle all mess
def get_IAMDataset_test(credentials):
    test_ds = IAMDataset("form_original", credentials=credentials, train=False)
    return test_ds


def select_device(device=None, num_device=1):
    if device is None:
        if num_device == 1:
            device_object = mx.gpu(0)
        else:
            device_object = [mx.gpu(i) for i in range(num_device)]
    elif device == 'auto':
        if num_device == 1:
            device_object = mx.gpu(num_device - 1) if mx.context.num_gpus() > 0 else mx.cpu(num_device - 1)
        else:
            device_object = [mx.gpu() if mx.context.num_gpus() > 0 else mx.cpu() for i in range(num_device)]
            # device = [mx.gpu(i) for i in range(num_device)] if mx.context.num_gpus() > 0 else [mx.cpu(i) for i in
            #                                                                                    range(num_device)]
    elif device == 'cpu':
        device_object = mx.cpu(num_device - 1)
    elif device == 'gpu':
        device_object = mx.gpu(num_device - 1)
    else:
        # If it isn't a string.
        print("Assuming device is a mxnet ctx or device object or queue. Exp: mx.gpu(0)")
        device_object = device

    return device_object


#         :param min_c: minimum probability of detected image
#         :param overlap_thres: overlapping constant
#         :param topk: number of maximum probability detected bounding boxes
#         :param show: Show histogram if show=True. default; show=False
class recognize:
    def __init__(self, image, form_size=(1120, 800), device=None, num_device=1, crop=False,
                 ScliteHelperPATH=None, show=False, is_test=False):
        """
        Handwritten Text Recognization in one step
        :param image: input image in numpy.array object that includes handwritten text
        :param form_size: possible form size
        :param device:
        If it is None:
            If num_device==1: uses gpu if there is any gpu
            else: uses num_device gpu if there is any gpu
        If it is 'auto':
            If num_device==1: uses gpu if there is any gpu
            else: uses num_device gpu if there is any gpu
        if it is 'cpu': uses one, num_device-1 indexed cpu
        if it is 'gpu': uses one, num_device-1 indexed gpu
        :param num_device: number of device that module running on.
        :param crop: cropping detected text area
        :param ScliteHelperPATH: Tool that helps to get quantitative results. https://github.com/usnistgov/SCTK
        :param show: Show plot if show=True. default; show=False
        """
        # TODO! make device selection in mxnet way like "device=mx.gpu(0)". It is better way to do it than strings.
        self.device = select_device(device, num_device)
        self.show = show
        self.crop = crop
        self.is_test = is_test
        self.form_size = form_size
        self.predicted_text_area = 0
        self.croped_image = 0
        self.segmented_paragraph_size = (700, 700)
        self.line_image_size = (60, 800)
        self.predicted_bb = 0
        self.min_c = 0.1
        self.overlap_thres = 0.1
        self.topk = 600
        self.line_images_array = []
        self.character_probs = []
        if ScliteHelperPATH is None:
            ScliteHelperPATH = '../SCTK/bin'
        if self.is_test:
            self.sclite = ScliteHelper(ScliteHelperPATH)

        download_models()

        assert type(image) is np.ndarray, "Please enter numpy array"
        self.image = image
        # self.image = mx.nd.array(image)  # converts to MXNet-NDarray
        # self.image = self.image.asnumpy()
        # loads with channels
        if len(self.image.shape) > 2:
            self.image = self.image[..., 0]  # converts gray scale
        # print(image.shape)
        # self.image = self.image[np.newaxis, :]  # add batch dim
        # # print(image.shape)

        # paragraph_segmentation_net
        self.paragraph_segmentation_net = SegmentationNetwork(ctx=self.device)
        self.paragraph_segmentation_net.cnn.load_parameters("models/paragraph_segmentation2.params", ctx=self.device)
        self.paragraph_segmentation_net.hybridize()
        print("Paragraph segmentation model loading")

        # word_segmentation_net
        self.word_segmentation_net = WordSegmentationNet(2, ctx=self.device)
        self.word_segmentation_net.load_parameters("models/word_segmentation2.params")
        self.word_segmentation_net.hybridize()
        print("Word segmentation model loading")

        ## Denoising the text output
        # We use a seq2seq denoiser to translate noisy input to better output
        self.FEATURE_LEN = 150
        self.denoiser = Denoiser(alphabet_size=len(ALPHABET), max_src_length=self.FEATURE_LEN,
                                 max_tgt_length=self.FEATURE_LEN,
                                 num_heads=16, embed_size=256, num_layers=2)
        self.denoiser.load_parameters('models/denoiser2.params', ctx=self.device)
        self.denoiser.hybridize(static_alloc=True)
        print("Denoiser model loading")

        ## We use a language model in order to rank the propositions from the denoiser
        self.language_model, self.vocab = nlp.model.big_rnn_lm_2048_512(dataset_name='gbw', pretrained=True,
                                                                        ctx=self.device)
        self.moses_tokenizer = nlp.data.SacreMosesTokenizer()
        self.moses_detokenizer = nlp.data.SacreMosesDetokenizer()

        # handwriting_line_recognition_net
        self.handwriting_line_recognition_net = HandwritingRecognitionNet(rnn_hidden_states=512,
                                                                          rnn_layers=2, ctx=self.device,
                                                                          max_seq_len=160)
        self.handwriting_line_recognition_net.load_parameters("models/handwriting_line8.params", ctx=self.device)
        self.handwriting_line_recognition_net.hybridize()
        print("Handwriting line segmentation model loading")

        ## We use beam search to sample the output of the denoiser
        self.beam_sampler = nlp.model.BeamSearchSampler(beam_size=20,
                                                        decoder=self.denoiser.decode_logprob,
                                                        eos_id=EOS,
                                                        scorer=nlp.model.BeamSearchScorer(),
                                                        max_length=150)
        print("Beam sampler created")
        self.generator = SequenceGenerator(self.beam_sampler, self.language_model, self.vocab, self.device,
                                           self.moses_tokenizer,
                                           self.moses_detokenizer)
        print("Sequence generator created")

    def __call__(self, *args, **kwargs):
        return self.one_step(*args, **kwargs)

    def one_step(self, expand_bb_scale_x=0.18, expand_bb_scale_y=0.23, segmented_paragraph_size=(700, 700)):
        """
        Calculate all in of them in one (long) step
        :param expand_bb_scale_x: Scale constant along x axis
        :param expand_bb_scale_y: Scale constant along y axis
        :param segmented_paragraph_size: segmented paragraph size in tuple
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
        predicted_text_area = self.predict_bbs(expand_bb_scale_x=expand_bb_scale_x, expand_bb_scale_y=expand_bb_scale_y)

        croped_image = None
        if self.crop:
            croped_image = self.crop_image(segmented_paragraph_size)

        predicted_bb = self.word_detection()
        line_images_array = self.word_to_line()
        character_probs = self.handwriting_recognition_probs()

        decoded = self.qualitative_result()
        # decoded_line_ams, decoded_line_bss, decoded_line_denoisers = decoded
        results = {
            'predicted_text_area': predicted_text_area,
            'croped_image': croped_image,
            'predicted_bb': predicted_bb,
            'line_images_array': line_images_array,
            'character_probs': character_probs,
            'decoded': decoded
        }
        return results

    ## Paragraph segmentation
    # Given the image of a form in the IAM dataset, predict a bounding box of the handwriten component. The model was trained on using https://github.com/ThomasDelteil/Gluon_OCR_LSTM_CTC/blob/master/paragraph_segmentation_dcnn.py and an example is presented in https://github.com/ThomasDelteil/Gluon_OCR_LSTM_CTC/blob/master/paragraph_segmentation_dcnn.ipynb
    def predict_bbs(self, expand_bb_scale_x=0.18, expand_bb_scale_y=0.23):
        """
        Predicts bounding box of given image to detect where the text in the image
        :param expand_bb_scale_x: Scale constant along x axis
        :param expand_bb_scale_y: Scale constant along y axis
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
    def word_detection(self):
        """
        Word detector with SSD(single shot detection)
        :return: predicted bounding box for each word
        """
        # paragraph_segmented_image = paragraph_segmented_images[0]
        paragraph_segmented_image = self.image
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
    def word_to_line(self):
        """
        Converts word bounding boxes to line bounding boxes by overlapping.
        :return: croped line images array.
        """
        paragraph_segmented_image = self.image

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
    def handwriting_recognition_probs(self):
        """
        Calculates character probabilities
        :param line_image_size: Posibble line size
        :return: Character probabilities
        """
        for line_images in self.line_images_array:
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
                    # print("[AM]", decoded_line_am)
                    decoded_line_ams.append(decoded_line_am)
                    decoded_line_bs = get_beam_search(line_character_probs)
                    decoded_line_bss.append(decoded_line_bs)
                    decoded_line_denoiser = self.get_denoised(line_character_probs.asnumpy(), ctc_bs=False)
                    # print("[D ]", decoded_line_denoiser)
                    decoded_line_denoisers.append(decoded_line_denoiser)
        else:
            for i, form_character_probs in enumerate(self.character_probs):
                fig, axs = plt.subplots(len(form_character_probs) + 1,
                                        figsize=(10, int(1 + 2.3 * len(form_character_probs))))
                for j, line_character_probs in enumerate(form_character_probs):
                    decoded_line_am = get_arg_max(line_character_probs)
                    print("[AM]", decoded_line_am)
                    decoded_line_ams.append(decoded_line_am)
                    decoded_line_bs = get_beam_search(line_character_probs)
                    decoded_line_bss.append(decoded_line_bs)
                    decoded_line_denoiser = self.get_denoised(line_character_probs.asnumpy(), ctc_bs=False)
                    print("[D ]", decoded_line_denoiser)
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
        print("Mean CER = {}".format(cer))
        return cer

    def get_qualitative_results(self, denoise_func, credentials=None):
        """
        Get all quantative "character error results" full pipeline
        :param credentials: credentials to access and download IAMdataset
        :param denoise_func: denoiser function
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
        print(CER0)
        CER.append(CER0)
        CER1 = self.get_qualitative_results_lines(self.get_denoised)
        print(CER1)
        CER.append(CER1)

        # %% md
        # CER full pipeline
        CER2 = self.get_qualitative_results(get_arg_max)
        print(CER2)
        CER.append(CER2)

        CER3 = self.get_qualitative_results(get_beam_search)
        print(CER3)
        CER.append(CER3)

        # %%
        cer_denoiser = self.get_qualitative_results(self.get_denoised)
        print(cer_denoiser)
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
        print(sentence)
        print("Choice")
        print(self.generator.generate_sequences(inputs, states, sentence))


## TEST
# Write test into this class
class recognize_test():
    def __init__(self, image_name="tests/TurkishHandwritten/elyaz2.jpeg", filter_number=1, form_size=(1120, 800),
                 device=None, num_device=1,
                 show=True):
        """
        Handwritten test initializer
        :param image_name: Image name with full path
            default; "elyaz2.jpeg"
        :param filter_number: There is 4 different filters 0-3
            default; 1
        :param form_size: poossible form size
            default; (1120, 800)
        :param device:
        If it is None:
            If num_device==1: uses gpu if there is any gpu
            else: uses num_device gpu if there is any gpu
        If it is 'auto':
            If num_device==1: uses gpu if there is any gpu
            else: uses num_device gpu if there is any gpu
        if it is 'cpu': uses one, num_device-1 indexed cpu
        if it is 'gpu': uses one, num_device-1 indexed gpu
        :param num_device: number of device that module running on.
        :param show: Show plot if show=True. default; show=False
        """
        self.form_size = form_size
        self.device = select_device(device, num_device)
        self.show = show
        # original image
        self.image = mx.image.imread(image_name)  # 0 is grayscale
        self.image = self.image.asnumpy()
        print(self.image.shape)

        self.image = self.resize()
        self.image = self.image[..., 0]

        self.images, self.titles = self.preprocess()
        # [img, th1, th2, th3] = images
        self.image = self.images[filter_number]
        print("filter number: ", filter_number, "filtered shape:", self.image.shape)

        # self.image = self.image[..., 0]
        self.htr = recognize(self.image, form_size=form_size, device=self.device, show=self.show)

    def __call__(self, *args, **kwargs):
        self.htr(*args, **kwargs)

    def histogram(self, show=False):
        """
        Gives histogram, normalized cdf and bins values in numpy type
        :param show: Show plot if show=True. default; show=False
        :return: histogram, normalized cdf and bins
        """
        return histogram(self.image, show=show or self.show)

    def resize(self, show=False):
        """
        Resizes numpy array into form size
        :param show: Show plot if show=True. default; show=False
        :return: Resized image in numpy format.
        """
        # TODO: MXNET gpu -> cpu
        # image = self.image.asnumpy()
        image = self.image
        image_resize = cv2.resize(image, dsize=self.form_size, interpolation=cv2.INTER_CUBIC)
        if show or self.show:
            fig, ax = plt.subplots(2, 1, figsize=(15, 18))
            print("test_image: ", image.shape)
            plt.subplot(121), plt.imshow(image, cmap="gray")
            plt.title("original image")
            plt.axis("off")

            # downsampled image
            print("test_HTR_downsampled: ", image_resize.shape)
            plt.subplot(122), plt.imshow(image_resize, cmap="gray")
            plt.title("downsampled image")
            plt.axis("off")

            # save downsampled test_image and review
            plt.imsave("HTR_downsampled.jpeg", image_resize, cmap="gray")
        return image_resize

    def preprocess(self, bottom=127, top=255, show=False):
        """
        Input image preprocess with some tricks.
        :param bottom: lower limit of filter.
        :param top: upper limit of filter.
        :param show: Show plot if show=True. default; show=False
        :return: filtered images and titles
        """
        at = all_togather(self.image, bottom, top)
        (th, laplacian, sobelx, sobely) = at
        [img, th1, th2, th3] = th
        titles = ['Original Image', 'Global Thresholding (v = 127)',
                  'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
        images = [img, th1, th2, th3]

        if show or self.show:
            fig, ax = plt.subplots(1, figsize=(15, 9))
            for i in range(4):
                plt.subplot(2, 2, i + 1), plt.imshow(images[i], 'gray')
                plt.title(titles[i])
                plt.xticks([]), plt.yticks([])
            plt.show()
            fig.savefig("elyaz_thresholds.png")
        return images, titles


class recognize_IAM_random_test():
    def __init__(self, device=None, num_device=1):
        self.device = select_device(device=device, num_device=num_device)
        test_ds = IAMDataset("form_original", train=False)

        n = random.random()  # random selection
        n = int(n * len(test_ds))
        self.image, self.text = test_ds[n]
        self.recog = recognize(self.image, device=device)

    def __call__(self, *args, **kwargs):
        return self.test()

    def test(self):
        result = self.recog()
        return result


class recognize_IAM_test():
    def __init__(self, num_image=4, device=None, num_device=1):
        self.device = select_device(device=device, num_device=num_device)
        test_ds_path_images = "tests/IAM8/"
        test_ds_images = [f for f in listdir(test_ds_path_images) if isfile(join(test_ds_path_images, f))]
        self.image_name = test_ds_images[num_image]
        self.image_path = test_ds_path_images + self.image_name

        test_ds_path_results = "tests/IAM8/results"
        test_ds_results = [f for f in listdir(test_ds_path_results) if isfile(join(test_ds_path_results, f))]
        self.text_name = test_ds_results[num_image]
        self.text_path = test_ds_path_results + self.text_name

        image = mx.image.imread(self.image_path)
        self.image = image.asnumpy()
        self.recog = recognize(self.image, device=device)

    def __call__(self, *args, **kwargs):
        return self.test()

    def test(self):
        result = self.recog()
        return result


if __name__ == "__main__":
    device = "cpu"

    # %% recognize_test class
    # htr_test = recognize_test(show=True, device=device)
    # result = htr_test()

    # %% recognize class
    # image = mx.image.imread("tests/TurkishHandwritten/elyaz2.jpeg")
    # image = image.asnumpy()
    # recog = recognize(image, device=device)
    # result = recog()

    # %% recognize_IAM_random_test class
    # IAM_recog = recognize_IAM_random_test(device)
    # result = IAM_recog()

    # %% recognize_IAM_test class
    IAM_recog = recognize_IAM_test(4, device)
    result = IAM_recog()

    pprint(result)
