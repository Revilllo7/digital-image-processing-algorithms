import os
import cv2
import numpy as np
import time

def otsu_threshold(hist):
    # works for histograms of any length
    n = hist.size
    total = hist.sum()
    if total == 0:
        return 0
    sumB = 0.0
    wB = 0.0
    maximum = -1.0
    # index vector matches histogram length
    idx = np.arange(n, dtype=float)
    sum1 = np.dot(idx, hist)

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
    return int(threshold)

def local_otsu_11x11(infile="resources/kwiatki.png", outdir="output"):
    img = cv2.imread(infile, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Input image not found: {infile}")

    # pad symmetrically (5 pixels on each side for 11x11 window)
    pad = 5
    padded = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REFLECT)

    out = np.zeros_like(img, dtype=np.uint8)
    h, w = img.shape

    for y in range(h):
        for x in range(w):
            window = padded[y:y+11, x:x+11]
            hist = cv2.calcHist([window], [0], None, [256], [0,256]).ravel()
            T = otsu_threshold(hist)
            out[y, x] = 255 if img[y, x] > T else 0

    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, "local_otsu_11x11.png")
    cv2.imwrite(out_path, out)
    print(f"Saved local Otsu result to: {out_path}")

    # optional: save side-by-side comparison (grayscale + segmentation colored)
    cmp = np.hstack((img, out))
    cv2.imwrite("output/comparison_11x11_kwiatki.png", cmp)

if __name__ == "__main__":
    t0 = time.perf_counter()
    local_otsu_11x11()
    print("Elapsed (s):", time.perf_counter() - t0)