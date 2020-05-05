import numpy as np
from matplotlib import pyplot as plt
import cv2


def histogram(image, show=False):
    """
    Gives histogram, normalized cdf and bins values in numpy type
    :param image: Image of numpy array.
    :param show: Show histogram if show=True. default; show=False
    :return: histogram, normalized cdf and bins
    """
    image = image.flatten()
    hist, bins = np.histogram(image, 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()
    if show:
        fig, ax = plt.subplots(1, figsize=(15, 9))
        plt.plot(cdf_normalized, color='b')
        plt.hist(image, 256, [0, 256], color='r')
        plt.xlim([0, 256])
        plt.legend(('cdf', 'histogram'), loc='upper left')
        plt.show()
    return hist, cdf_normalized, bins


def thresholds(image, bottom=127, top=255):
    img = cv2.medianBlur(image, 5)
    ret, th1 = cv2.threshold(img, bottom, top, cv2.THRESH_TOZERO + cv2.THRESH_OTSU)  # Global thresholding
    th2 = cv2.adaptiveThreshold(img, top, cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY, 11, 2)
    th3 = cv2.adaptiveThreshold(img, top, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 11, 2)
    return img, th1, th2, th3


def equalizeHist(image):
    return cv2.equalizeHist(image)


# create a CLAHE object (Arguments are optional).
def CLAHE(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl1 = clahe.apply(img)
    return cl1


def filters(image):
    # kernel
    kernel_size = 21  # odd number, 5
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=kernel_size)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=kernel_size)
    return laplacian, sobelx, sobely


def all_togather(image, bottom=127, top=255):
    """
    Preprocess combination.
    :param image: a numpy array
    :param bottom:
    :param top:
    :return:
    """
    # CLAHE
    cl1 = CLAHE(image)
    th = thresholds(cl1, bottom, top)
    [image, th1, th2, th3] = th

    # kernel
    kernel_size = 21  # odd number, 5
    laplacian = cv2.Laplacian(th1, cv2.CV_64F)
    sobelx = cv2.Sobel(th1, cv2.CV_64F, 1, 0, ksize=kernel_size)
    sobely = cv2.Sobel(th1, cv2.CV_64F, 0, 1, ksize=kernel_size)

    at = (th, laplacian, sobelx, sobely)
    return at


if __name__ == "__main__":
    pass
