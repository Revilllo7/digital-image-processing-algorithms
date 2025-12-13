import numpy as np
from scipy.ndimage import generic_filter
from imageio import imread, imwrite

def mid_range_filter(window):
    return (window.min() + window.max()) / 2


img = imread("resources/Jellyfish.png").astype(np.float32)

out = generic_filter(img, mid_range_filter, size=3, mode="reflect")

imwrite("output/c_midrange.png", np.clip(out,0,255).astype(np.uint8))