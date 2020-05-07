# TODO! Not Completed!
import os
import random
from os import listdir, walk
from os.path import isfile, join
import glob
from pprint import pprint

import cv2
import matplotlib.pyplot as plt
import mxnet as mx

from recognition.ocr.utils.iam_dataset import IAMDataset
from recognition.ocr.utils.preprocess import histogram, all_togather
from recognition.recognizer import select_device, recognize


class recognize_test():
    """
    Handwritten recognition test with one particular image in numpy type
    """

    def __init__(self, image_name="TurkishHandwritten/elyaz2.jpeg", filter_number=1, form_size=(1120, 800),
                 device=None, num_device=1,
                 show=True):
        """
        Handwritten recognition test initializer
        :param image_name: Image name with full path
            DEFAULT="elyaz2.jpeg"
        :param filter_number: There is 4 different filters 0-3
            DEFAULT=1
        :param form_size: poossible form size
            DEFAULT=(1120, 800)
        :param device:
        If it is None:
            If num_device==1: uses gpu if there is any gpu
            else: uses num_device gpu if there is any gpu
        If it is 'auto':
            If num_device==1: uses gpu if there is any gpu
            else: uses num_device gpu if there is any gpu
        if it is 'cpu': uses one, num_device-1 indexed cpu
        if it is 'gpu': uses one, num_device-1 indexed gpu
            DEFAULT=None
        :param num_device: number of device that module running on.
            DEFAULT=1
        :param show: Show plot if show=True. default; show=False
            DEFAULT=True
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

    def __init__(self, device=None, num_device=1):
        """
        Handwritten recognition random image test initializer
        :param device:
        If it is None:
            If num_device==1: uses gpu if there is any gpu
            else: uses num_device gpu if there is any gpu
        If it is 'auto':
            If num_device==1: uses gpu if there is any gpu
            else: uses num_device gpu if there is any gpu
        if it is 'cpu': uses one, num_device-1 indexed cpu
        if it is 'gpu': uses one, num_device-1 indexed gpu
            DEFAULT=None
        :param num_device: number of device that module running on.
            DEFAULT=1
        """
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
    """
    Handwritten recognition test select from subset of IAM dataset. images at IAM8 folder
    """

    def __init__(self, net_parameter_names, num_image=4, device=None, num_device=1):
        """
        Test recognize class with subset of IAM dataset. Images and predicted results are in the 'IAM8' folder.
        :param num_image: image number. 1-8
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
        """
        assert not (num_image < 1 or num_image > 8), "Please enter number between 1-8"
        self.device = select_device(device=device, num_device=num_device)
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

        self.recog = recognize(self.image, net_parameters=net_parameter_names, device=device)

        f = []
        for (dirpath, dirnames, filenames) in walk(net_parameter_path):
            f.extend(filenames)
            break

    def __call__(self, *args, **kwargs):
        return self.test()

    def test(self):
        result = self.recog()
        return result


if __name__ == "__main__":
    net_parameter_path = "../models"
    net_module_paths = [f for f in listdir(net_parameter_path) if isfile(join(net_parameter_path, f))]
    net_parameter_names = [param for param in net_module_paths if os.path.splitext(param)[1] == ".params"]
    net_parameter_paths = [os.path.join(net_parameter_path, path) for path in net_parameter_names]

    device = "cpu"

    # %% recognize_test class
    # htr_test = recognize_test(show=True, device=device)
    # result = htr_test()

    # %% recognize class
    # image = mx.image.imread("TurkishHandwritten/elyaz2.jpeg")
    # image = image.asnumpy()
    # recog = recognize(image, device=device)
    # result = recog()

    # %% recognize_IAM_random_test class
    # IAM_recog = recognize_IAM_random_test(device)
    # result = IAM_recog()

    # %% recognize_IAM_test class
    IAM_recog = recognize_IAM_test(net_parameter_paths, 4, device)
    result = IAM_recog()

    pprint(result)
