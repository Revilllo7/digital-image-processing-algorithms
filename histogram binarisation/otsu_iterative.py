import cv2
import numpy as np

img = cv2.imread("resources/kwiatki.png", cv2.IMREAD_GRAYSCALE)

def threshold_otsu(image):
    hist, _ = np.histogram(image.flatten(), bins=256, range=[0,256])
    total = image.size
    current_max, threshold = 0, 0
    sum_total, sum_foreground = 0, 0
    weight_background, weight_foreground = 0, 0

    for i in range(256):
        sum_total += i * hist[i]

    for i in range(256):
        weight_background += hist[i]
        if weight_background == 0:
            continue
        weight_foreground = total - weight_background
        if weight_foreground == 0:
            break
        sum_foreground += i * hist[i]
        mean_background = sum_foreground / weight_background
        mean_foreground = (sum_total - sum_foreground) / weight_foreground
        between_class_variance = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2
        if between_class_variance > current_max:
            current_max = between_class_variance
            threshold = i
    return threshold

def iterative_otsu(img, delta = 2):
    T_prev = threshold_otsu(img)
    while True:
        lower = img[img <= T_prev]
        upper = img[img > T_prev]
        if len(lower) == 0 or len(upper) == 0:
            break
        mu0, mu1 = np.mean(lower), np.mean(upper)
        subset = img[(img > mu0) & (img <= mu1)]
        if subset.size == 0:
            break
        T_new = threshold_otsu(subset)
        if abs(T_new - T_prev) < delta:
            break
        T_prev = T_new
    return T_prev

T_iterative = iterative_otsu(img)

result_iterative = np.zeros_like(img)
result_iterative[img >= T_iterative] = 255
result_iterative[(img < T_iterative) & (img >= T_iterative/2)] = 127

# --- added: print thresholds and save outputs ---
print(f"Computed threshold: T_iterative={T_iterative}")
cv2.imwrite("output/segmented_kwiatki.png", result_iterative)

# optional: save side-by-side comparison (grayscale + segmentation colored)
cmp = np.hstack((img, result_iterative))
cv2.imwrite("output/comparison_segmented_kwiatki.png", cmp)