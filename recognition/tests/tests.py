# TODO! Not Completed!
import os
import random
from os import listdir
from os.path import isfile, join
import logging

from pprint import pprint

import cv2
import matplotlib.pyplot as plt
import mxnet as mx

try:
    from recognition.ocr.utils.iam_dataset import IAMDataset
    from recognition.ocr.utils.preprocess import histogram, all_togather
    from recognition.recognizer import recognize
    from recognition.utils.recognizer_utils import device_selection_helper
except:
    try:
        from handwritten_text_recognition.recognition.ocr.utils.iam_dataset import IAMDataset
        from handwritten_text_recognition.recognition.ocr.utils.preprocess import histogram, all_togather
        from handwritten_text_recognition.recognition.recognizer import recognize
        from handwritten_text_recognition.recognition.utils.recognizer_utils import device_selection_helper
    except:
        from recognition.handwritten_text_recognition.recognition.ocr.utils.iam_dataset import IAMDataset
        from recognition.handwritten_text_recognition.recognition.ocr.utils.preprocess import histogram, all_togather
        from recognition.handwritten_text_recognition.recognition.recognizer import recognize
        from recognition.handwritten_text_recognition.recognition.utils.recognizer_utils import device_selection_helper

## TEST
# Write test into this class
class recognize_test():
    """
    Handwritten recognition test with one particular image in numpy type
    """

    def __init__(self, image_name="TurkishHandwritten/elyaz2.jpeg", net_parameter_pathname=None, filter_number=1, form_size=(1120, 800), device=None,
                 show=True):
        """
        Handwritten recognition test initializer
        :param image_name: Image name with full path
            DEFAULT="elyaz2.jpeg"
        :param net_parameter_pathname: network(model) parameter paths with name
        :param filter_number: There is 4 different filters 0-3
            DEFAULT=1
        :param form_size: poossible form size
            DEFAULT=(1120, 800)
        :param device: determines the device that the model(or network) will work on
            DEFAULT=None
        If it is None:
            device = mx.cpu()
        :param show: Show plot if show=True. default; show=False
            DEFAULT=True
        """
        self.form_size = form_size
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
        self.htr = recognize(self.image, net_parameter_pathname=net_parameter_pathname, form_size=form_size, device=device, show=self.show)

    def __call__(self, *args, **kwargs):
        self.htr(*args, **kwargs)

    def histogram(self, show=False):
        """
        Gives histogram, normalized cdf and bins values in numpy type
        :param show: Show plot if show=True. default; show=False
            DEFAULT=False
        :return: histogram, normalized cdf and bins
        """
        return histogram(self.image, show=show or self.show)

    def resize(self, show=False):
        """
        Resizes numpy array into form size
        :param show: Show plot if show=True. default; show=False
            DEFAULT=False
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
            DEFAULT=127
        :param top: upper limit of filter.
            DEFAULT=255
        :param show: Show plot if show=True. default; show=False
            DEFAULT=False
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
    """
    Handwritten recognition test select randomly from IAM dataset
    """

    def __init__(self, net_parameter_pathname, device=None):
        """
        Handwritten recognition random image test initializer
        :param net_parameter_pathname: network(model) parameter paths with name
        :param device: determines the device that the model(or network) will work on
            DEFAULT=None
        If it is None:
            device = mx.cpu()
        """
        test_ds = IAMDataset("form_original", train=False)

        n = random.random()  # random selection
        n = int(n * len(test_ds))
        self.image, self.text = test_ds[n]
        self.recog = recognize(self.image, net_parameter_pathname=net_parameter_pathname, device=device)

    def __call__(self, *args, **kwargs):
        return self.test()

    def test(self):
        result = self.recog()
        return result


class recognize_IAM_test():
    """
    Handwritten recognition test select from subset of IAM dataset. images at IAM8 folder
    """

    def __init__(self, net_parameter_pathname, num_image=4, device=None):
        """
        Test recognize class with subset of IAM dataset. Images and predicted results are in the 'IAM8' folder.
        :param net_parameter_pathname: network(model) parameter paths with name
        :param num_image: image number. 1-8
        :param device: determines the device that the model(or network) will work on
            DEFAULT=None
        If it is None:
            device = mx.cpu()
        """
        assert not (num_image < 1 or num_image > 8), "Please enter number between 1-8"
        test_ds_path_images = "IAM8/"
        test_ds_images = [f for f in listdir(test_ds_path_images) if isfile(join(test_ds_path_images, f))]
        self.image_name = test_ds_images[num_image]
        self.image_path = test_ds_path_images + self.image_name

        test_ds_path_results = "IAM8/results"
        test_ds_results = [f for f in listdir(test_ds_path_results) if isfile(join(test_ds_path_results, f))]
        self.text_name = test_ds_results[num_image]
        self.text_path = test_ds_path_results + self.text_name

        image = mx.image.imread(self.image_path)
        self.image = image.asnumpy()

        self.recog = recognize(self.image, net_parameter_pathname=net_parameter_pathname, device=device)

    def __call__(self, *args, **kwargs):
        return self.test()

    def test(self):
        result = self.recog()
        return result


if __name__ == "__main__":
    # TODO! models path string isn't looks good.
    net_parameter_path = r"../models"
    net_module_paths = [f for f in listdir(net_parameter_path) if isfile(join(net_parameter_path, f))]
    net_parameter_names = [param for param in net_module_paths if os.path.splitext(param)[1] == ".params"]
    net_parameter_paths = [os.path.join(net_parameter_path, path) for path in net_parameter_names]

    num_device = 1
    device_queue = "cpu"
    device = device_selection_helper(device=device_queue, num_device=num_device)

    import time
    t0 = time.time()

    # # %% recognize_test class
    # htr_test = recognize_test(show=True, net_parameter_pathname=net_parameter_paths, device=device)
    # result = htr_test()

    # # %% recognize class
    # image = mx.image.imread("TurkishHandwritten/elyaz2.jpeg")
    # image = image.asnumpy()
    # recog = recognize(image, net_parameter_paths, device=device)
    # result = recog()

    # %% recognize_IAM_random_test class
    IAM_recog = recognize_IAM_random_test(net_parameter_pathname=net_parameter_paths, device=device)
    result = IAM_recog()

    # # %% recognize_IAM_test class
    # IAM_recog = recognize_IAM_test(net_parameter_pathname=net_parameter_paths, num_image=4, device=device)
    # result = IAM_recog()

    pprint(result)
    print("Elapsed time", time.time()-t0)
