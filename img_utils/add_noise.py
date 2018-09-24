import cv2
import numpy as np
import os
import random


def add_gaussian_noise(image, alpha=0.9):
    row, col, ch = image.shape
    mean = 0
    var = 0.1
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    noisy_img = alpha * image + (1 - alpha) * gauss
    return noisy_img