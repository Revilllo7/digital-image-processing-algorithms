import numpy as np
from scipy.ndimage import generic_filter
from imageio import imread, imwrite

def k_nearest_neighbor(window, k=6):
    center = window[4]
    distances = np.abs(window - center)
    idx = np.argsort(distances)[:k]
    return window[idx].mean()


img = imread("resources/Jellyfish.png").astype(np.float32)

out = generic_filter(img, k_nearest_neighbor, size=3, mode="reflect")

imwrite("output/d_knn.png", np.clip(out,0,255).astype(np.uint8))