import numpy as np
from scipy.ndimage import generic_filter
from imageio import imread, imwrite

def snn_filter(window):
    center = window[4]
    pairs = [(0,8), (1,7), (2,6), (3,5)]
    selected = [center]

    for i, j in pairs:
        if abs(window[i] - center) <= abs(window[j] - center):
            selected.append(window[i])
        else:
            selected.append(window[j])

    return np.mean(selected)


img = imread("resources/Jellyfish.png").astype(np.float32)

out = generic_filter(img, snn_filter, size=3, mode="reflect")

imwrite("output/e_snn.png", np.clip(out,0,255).astype(np.uint8))