"""
This module contains all necessary functions for pre-processing activity of
images. The module contains general functions for pre-processing images and preparing
data for training and testing regression model.
"""

import os

import PIL
import numpy
import matplotlib


def get_image_vector(img_path, dims):
    """
    This function gets an image's path and dimensions and than returns the resized(by the
    given dimensions) image's raw vector(based on it's pixels normalized to 255).
    :param img_path: The path of the image we want to preprocess.
    :type img_path: C{str}
    :param dims: The dimensions we want to turn the image to. to unify it to single NN.
    :type dims: c{tuple}
    :return: A 1D vector which contains the image's pixels data normalized to 255.
    :rtype: C{np.array}
    """

    img_arr = numpy.array(matplotlib.image.mpimg.imread(img_path, img_path.split(".")[-1]))
    img = PIL.Image.fromarray(img_arr)
    img = img.resize(dims, PIL.Image.ANTIALIAS)
    img_vec = numpy.array(img)
    img_vec = img_vec / 255
    img_vec = img_vec.reshape((dims[0]*dims[1]*3),)
    return img_vec


def get_dir_images_vector_list(dir_path, dims):
    """
    This function makes an array af the images' vectors that are stored in the current directory.
    The main benefit of that function is in the creation of Train X vector.
    :param dir_path: The path of a dir with images to turn to a vector.
    :type dir_path: C{str}
    :param dims: The desire dimensions of the images we are turning to vectors.
    :type dims:C{tuple}
    :return: An list(almost vector) of the directory's images' vectors.
    :rtype C{list}
    """

    # The array we are going to return.
    imgs_vecs = []
    images_file_names = os.listdir(dir_path)
    for img_name in images_file_names:
        imgs_vecs.append(get_image_vector(f"{dir_path}/{img_name}", dims))

    return imgs_vecs


def load_dataset():
    raise NotImplementedError()


if __name__ == '__main__':
    print(os.listdir("../venv"))
