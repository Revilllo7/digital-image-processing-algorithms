import argparse
import os
import numpy as np
from skimage import io, img_as_float
from scipy.ndimage import generic_filter
import matplotlib.pyplot as plt
import matplotlib as mpl

def _local_contrast_window(window):
    # Local contrast for center pixel in a NxN window: max - min
    return np.max(window) - np.min(window)

def compute_local_contrast(image, size=3):
    # Compute local contrast map using an NxN neighborhood (default 3x3).
    return generic_filter(image, _local_contrast_window, size=size)

def compute_lc_stats(lc_map):
    # Return basic statistics for the local contrast map (floats in [0,1]).
    mn = float(np.min(lc_map))
    mx = float(np.max(lc_map))
    mean = float(np.mean(lc_map))
    median = float(np.median(lc_map))
    std = float(np.std(lc_map))
    return {"min": mn, "max": mx, "mean": mean, "median": median, "std": std}

def scale_to_255(value):
    # Scale a float in [0,1] to integer in [0,255]
    return int(np.clip(round(value * 255.0), 0, 255))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute local contrast map (8-connected).")
    parser.add_argument("image", nargs="?", default="resources/lion_b.png", help="Path to grayscale image")
    parser.add_argument("--size", type=int, default=3, help="Neighborhood size (odd integer)")
    parser.add_argument("--out", default=None, help="Output path for saved figure (if not interactive)")
    args = parser.parse_args()

    img = img_as_float(io.imread(args.image, as_gray=True))
    lc_map = compute_local_contrast(img, size=args.size)

    # Compute and print numeric stats for the local contrast map
    stats = compute_lc_stats(lc_map)
    s_min = scale_to_255(stats["min"])
    s_max = scale_to_255(stats["max"])
    s_mean = scale_to_255(stats["mean"])
    s_median = scale_to_255(stats["median"])
    s_std = int(round(stats["std"] * 255.0))

    print("Local contrast stats (float 0..1): "
          f"min={stats['min']:.6f}, max={stats['max']:.6f}, mean={stats['mean']:.6f}, "
          f"median={stats['median']:.6f}, std={stats['std']:.6f}")
    print("Local contrast stats (scaled 0..255): "
          f"min={s_min}, max={s_max}, mean≈{s_mean}, median={s_median}, std≈{s_std}")

    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.imshow(img, cmap="gray")
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1,2,2)
    # Show local contrast in grayscale (values in [0,1])
    plt.imshow(lc_map, cmap="gray", vmin=0, vmax=1)
    plt.title(f"Local contrast (size={args.size}x{args.size})")
    plt.axis("off")

    # If backend is interactive, show; otherwise save to file to avoid FigureCanvasAgg warning.
    backend = mpl.get_backend()
    if backend in mpl.rcsetup.interactive_bk:
        plt.show()
    else:
        out_path = args.out
        if not out_path:
            base = os.path.splitext(args.image)[0].replace("/", "_").replace("\\", "_")
            out_path = f"output/{base}_local_contrast_size{args.size}.png"
        plt.savefig(out_path, bbox_inches='tight', dpi=150)
        print(f"Saved figure to {out_path} (backend: {backend})")