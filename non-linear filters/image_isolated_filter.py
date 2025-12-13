import numpy as np
from scipy.ndimage import generic_filter
from imageio import imread, imwrite

def isolated_pixel_filter(window):
    center = window[4]
    neighbors = np.delete(window, 4)
    mu = neighbors.mean()
    theta = 0.1 * mu
    return mu if abs(center - mu) > theta else center

img = imread("resources/Jellyfish.png").astype(np.float32)

out = generic_filter(img, isolated_pixel_filter, size=3, mode="reflect")

imwrite("output/a_isolated_removed.png", np.clip(out,0,255).astype(np.uint8))