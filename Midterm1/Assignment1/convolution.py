import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import math
from math import pi

        
def convolution(image: np.array, kernel : np.array, pad=True) -> np.array:
    # get width and height of the kernel
    k_w, k_h = kernel.shape

    # check if the image is in gray scale or RGB
    if (len(image.shape) == 2):
        # if the image is in gray scale we add the channel dimension
        image = image.reshape(tuple(list(image.shape) + [1]))

    w, h, c = image.shape

    # check if the image needs to be padded or not
    if pad:
        w, h = (w + k_w - 1, h + k_w - 1)
        padded = np.zeros(shape=(w, h, c))
        padded[k_w//2:k_w//2+w-k_w+1, k_h//2:k_h//2+h-k_h+1, :] = image
        image = padded

    out = np.zeros(shape=(w-k_w+1, h-k_h+1, c))
    for i in range(w-k_w):
        for j in range(h-k_h):
            for z in range(c):
                out[i, j, z] = np.sum(np.multiply(image[i:i+k_w, j:j+k_h, z], kernel))
    
    if out.shape[2] == 1:
        out = out.reshape(out.shape[:-1])

    return out


def convolution_gray_scale(image: np.array, kernel : np.array, pad=True) -> np.array:
    k_w, k_h = kernel.shape
    w, h = image.shape

    if pad:
        w, h = (w + k_w, h + k_w)
        padded = np.zeros(shape=(w, h))
        padded[k_w//2:w-k_w//2-1, k_h//2:h-k_w//2-1] = image
        image = padded

    out = np.zeros(shape=(w-k_w, h-k_h))
    for i in range(w-k_w):
        for j in range(h-k_h):
            out[i, j] = np.sum(np.multiply(image[i:i+k_w, j:j+k_h], kernel))
    return out


def gaussian_filter(shape=(3,3), sigma=1.0) -> np.array:
    m, n = [(ss-1.)/2. for ss in shape]
    y, x = np.ogrid[-m:m+1, -n:n+1]
    return 1/(2*math.pi*sigma*sigma)*np.exp(-(x*x + y*y) / (2.*sigma*sigma))


def log_filter(shape=(3,3), sigma=1.0) -> np.array:
    # w, h = shape
    # y, x = np.ogrid[-(w-1)/2:w/2, -(h-1)/2:h/2]
    m, n = [(ss-1.)/2. for ss in shape]
    y, x = np.ogrid[-m:m+1, -n:n+1]
    sigma_2 = sigma*sigma
    z = -(x*x + y*y)/(2*sigma_2)
    return -(1/(sigma_2*sigma_2))*(1+z)*np.exp(z)*sigma_2


def get_grid(shape=(3,3)):
    w, h = shape
    return np.meshgrid(np.linspace(-(w-1)/2,(w-1)/2, w), np.linspace(-(h-1)/2,(h-1)/2, h))


def gaussian_kernel(shape=(3,3), sigma=1.0) -> np.array:
    x, y = get_grid(shape)
    kernel = np.exp(-(x*x+y*y)/(2*sigma*sigma))
    return kernel / np.sum(kernel)


def log_kernel(shape=(3,3), sigma=1.0) -> np.array:
    x, y = get_grid(shape)
    sigma_2 = sigma*sigma
    z = -(x*x + y*y)/(2*sigma_2)
    return -(1/(sigma_2*sigma_2))*(1+z)*np.exp(z)*sigma_2
