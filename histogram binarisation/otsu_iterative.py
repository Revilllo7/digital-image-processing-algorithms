import cv2
import numpy as np

img = cv2.imread("resources/kwiatki.png", cv2.IMREAD_GRAYSCALE)

def otsu_threshold(hist):
    # handle histogram of any length (e.g. when using slices)
    n = hist.size
    total = hist.sum()
    sumB = 0.0
    wB = 0.0
    maximum = -1.0
    sum1 = np.dot(np.arange(n), hist)

    threshold = 0
    for t in range(n):
        wB += hist[t]
        if wB == 0:
            continue
        wF = total - wB
        if wF == 0:
            break
        sumB += t * hist[t]
        mB = sumB / wB
        mF = (sum1 - sumB) / wF
        between = wB * wF * (mB - mF) ** 2
        if between > maximum:
            maximum = between
            threshold = t
    return threshold

def iterative_three_class_threshold(img, delta_limit=2, max_iters=256):
    hist = cv2.calcHist([img],[0],None,[256],[0,256]).ravel()

    # initial thresholds
    T1_old, T2_old = 60, 180

    it = 0
    while True:
        it += 1
        # compute thresholds for each class using Otsu
        T1_new = otsu_threshold(hist[:T1_old+1])
        T2_new = otsu_threshold(hist[T1_old+1:]) + T1_old + 1

        if abs(T1_new - T1_old) < delta_limit and abs(T2_new - T2_old) < delta_limit:
            break

        if it >= max_iters:
            # stop to avoid infinite loop
            print(f"Warning: reached max_iters={max_iters}, returning current thresholds")
            break

        T1_old, T2_old = T1_new, T2_new

    return T1_new, T2_new

T1, T2 = iterative_three_class_threshold(img)

# trójklasowa segmentacja
seg = np.zeros_like(img)
seg[img > T1] = 127
seg[img > T2] = 255

# --- added: print thresholds and save outputs ---
print(f"Computed thresholds: T1={T1}, T2={T2}")
cv2.imwrite("output/segmented_kwiatki.png", seg)

# optional: save side-by-side comparison (grayscale + segmentation colored)
cmp = np.hstack((img, seg))
cv2.imwrite("output/comparison_segmented_kwiatki.png", cmp)