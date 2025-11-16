import os
import time
import cv2
import numpy as np

def otsu_threshold(hist):
    n = hist.size
    total = hist.sum()
    if total == 0:
        return 0
    sumB = 0.0
    wB = 0.0
    maximum = -1.0
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

def dynamic_otsu(img, iterations=3):
    result = img.copy().astype(np.uint8)
    for i in range(iterations):
        hist = cv2.calcHist([result],[0],None,[256],[0,256]).ravel()
        T = otsu_threshold(hist)
        # keep pixels above threshold, zero others
        result = np.where(result > T, result, 0).astype(np.uint8)
        print(f"iter {i+1}/{iterations}: T={T}")
    return result

def main(infile="resources/kwiatki.png", outdir="output", iterations=3):
    img = cv2.imread(infile, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Input image not found: {infile}")

    t0 = time.perf_counter()
    dyn = dynamic_otsu(img, iterations=iterations)
    elapsed = time.perf_counter() - t0

    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, "dynamic_otsu_kwiatki.png")
    cv2.imwrite(out_path, dyn)

    # side-by-side comparison (grayscale + dynamic result)
    cmp = np.hstack((img, dyn))
    cmp_path = os.path.join(outdir, "comparison_dynamic_kwiatki.png")
    cv2.imwrite(cmp_path, cmp)

    print(f"Saved dynamic Otsu: {out_path}")
    print(f"Saved comparison:    {cmp_path}")
    print(f"Elapsed (s): {elapsed:.3f}")

if __name__ == "__main__":
    main()