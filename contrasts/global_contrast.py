import argparse
import numpy as np
from skimage import io, img_as_float

def global_range_numpy(image):
    # Return (min, max, range) using NumPy.
    mn = np.min(image)
    mx = np.max(image)
    return float(mn), float(mx), float(mx - mn)

def global_range_iterative(image):
    # Return (min, max, range) using explicit iteration.
    h, w = image.shape
    min_val = float("inf")
    max_val = float("-inf")
    for y in range(h):
        for x in range(w):
            v = float(image[y, x])
            if v < min_val:
                min_val = v
            if v > max_val:
                max_val = v
    return float(min_val), float(max_val), float(max_val - min_val)

def scale_to_255(value):
    # Scale a float in [0,1] to integer in [0,255]
    return int(np.clip(round(value * 255.0), 0, 255))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute global min/max/range of a grayscale image.")
    parser.add_argument("image", nargs="?", default="resources/lion_c.png", help="Path to grayscale image")
    args = parser.parse_args()

    img = img_as_float(io.imread(args.image, as_gray=True))
    mn, mx, rng = global_range_numpy(img)
    mn_it, mx_it, rng_it = global_range_iterative(img)

    # Print float results (0..1)
    print(f"NumPy -> min: {mn:.6f}, max: {mx:.6f}, range: {rng:.6f}")
    print(f"Iter  -> min: {mn_it:.6f}, max: {mx_it:.6f}, range: {rng_it:.6f}")

    # Print scaled results (0..255)
    s_mn, s_mx = scale_to_255(mn), scale_to_255(mx)
    s_mn_it, s_mx_it = scale_to_255(mn_it), scale_to_255(mx_it)
    s_rng, s_rng_it = s_mx - s_mn, s_mx_it - s_mn_it

    print(f"Scaled (0-255) -> min: {s_mn}, max: {s_mx}, range: {s_rng}")
    print(f"Scaled Iter (0-255) -> min: {s_mn_it}, max: {s_mx_it}, range: {s_rng_it}")