import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import math
from skimage import color
from skimage import io

def load(image_path):
    """ Loads an image from a file path

    Args:
        image_path: file path to the image

    Returns:
        out: numpy array of shape(image_height, image_width, 3)
    """
    out = None

    ### YOUR CODE HERE
    # Use skimage io.imread
    out = io.imread(image_path)
    ### END YOUR CODE

    return out


def change_value(image):
    """ Change the value of every pixel by following x_n = 0.5*x_p^2 
        where x_n is the new value and x_p is the original value

    Args:
        image: numpy array of shape(image_height, image_width, 3)

    Returns:
        out: numpy array of shape(image_height, image_width, 3)
    """

    out = None

    ### YOUR CODE HERE
    out = 0.5 * np.square(image)
    ### END YOUR CODE

    return out


def convert_to_grey_scale(image):
    """ Change image to gray scale

    Args:
        image: numpy array of shape(image_height, image_width, 3)

    Returns:
        out: numpy array of shape(image_height, image_width, 3)
    """
    out = None

    ### YOUR CODE HERE
    out = np.sum(image * np.array([3, 6, 1]) / 10, axis=2)
    ### END YOUR CODE

    return out

def rgb_decomposition(image, channel):
    """ Return image **excluding** the rgb channel specified

    Args:
        image: numpy array of shape(image_height, image_width, 3)
        channel: str specifying the channel

    Returns:
        out: numpy array of shape(image_height, image_width, 3)
    """

    out = None

    ### YOUR CODE HERE
    dic = {'r':0, 'g':1, 'b':2}
    mat = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=np.float32)
    rgb = np.array(image)
    out = rgb * mat[dic[channel.lower()]]
    ### END YOUR CODE

    return out

def lab_decomposition(image, channel):
    """ Return image decomposed to just the lab channel specified

    Args:
        image: numpy array of shape(image_height, image_width, 3)
        channel: str specifying the channel

    Returns:
        out: numpy array of shape(image_height, image_width, 3)
    """

    lab = color.rgb2lab(image)
    out = None

    ### YOUR CODE HERE
    dic = {'l':0, 'a':1, 'b':2}
    mat = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=np.float32)
    out = lab * mat[dic[channel.lower()]]
    ### END YOUR CODE

    return out

def hsv_decomposition(image, channel='H'):
    """ Return image decomposed to just the hsv channel specified

    Args:
        image: numpy array of shape(image_height, image_width, 3)
        channel: str specifying the channel

    Returns:
        out: numpy array of shape(image_height, image_width, 3)
    """

    hsv = color.rgb2hsv(image)
    out = None

    ### YOUR CODE HERE
    dic = {'h':0, 's':1, 'v':2}
    mat = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=np.float32)
    out = hsv * mat[dic[channel.lower()]]
    ### END YOUR CODE

    return out

def mix_images(image1, image2, channel1, channel2):
    """ Return image which is the left of image1 and right of image 2 excluding
    the specified channels for each image

    Args:
        image1: numpy array of shape(image_height, image_width, 3)
        image2: numpy array of shape(image_height, image_width, 3)
        channel1: str specifying channel used for image1
        channel2: str specifying channel used for image2

    Returns:
        out: numpy array of shape(image_height, image_width, 3)
    """

    out = None
    ### YOUR CODE HERE
    image1 = rgb_decomposition(image1, channel1)
    image2 = rgb_decomposition(image2, channel2)
    out = np.zeros_like(image1)
    out[:, :image1.shape[1]/2, :] = image1[:, :image1.shape[1]/2, :]
    out[:, image1.shape[1]/2:, :] = image2[:, image1.shape[1]/2:, :]
    ### END YOUR CODE

    return out
