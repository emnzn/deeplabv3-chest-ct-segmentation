import os
import cv2
import numpy as np
from typing import Tuple

def compute_stats(img_dir: str) -> Tuple[float, float]:
    """
    Computes the pixel-level mean and standard deviation across all training images.
    
    Parameters
    ----------
    img_dir: str
        The parent directory containing all training images.

    Returns
    -------
    mean: float
        The mean pixel value of all images in the training set for normalization.
    
    std: float
        The pixel-level standard deviation of all images in the training set for normalization.
    """

    img_paths = [os.path.join(img_dir, img_name) for img_name in os.listdir(img_dir)]
    total_sum = 0
    total_num_pixels = 0
    total_sq_sum = 0

    for img_path in img_paths:
        img = cv2.imread(img_path, 0) / 255
        total_sum += img.sum()
        total_num_pixels += img.size
        total_sq_sum += (img ** 2).sum()

    mean = total_sum / total_num_pixels
    std = np.sqrt((total_sq_sum / total_num_pixels) - (mean ** 2))

    return mean, std