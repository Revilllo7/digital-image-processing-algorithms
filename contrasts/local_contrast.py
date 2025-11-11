import argparse
import os
import numpy as np
from skimage import io, img_as_float
from scipy.ndimage import generic_filter
import matplotlib.pyplot as plt
import matplotlib as mpl

def _local_contrast_window(window):
    # Local contrast as mean absolute difference between center and neighbors
    center = window[len(window)//2]
    neighbors = np.delete(window, len(window)//2)
    g_nb = np.mean(neighbors)
    return abs(center - g_nb)

def compute_local_contrast(image, size=3):
    # Compute local contrast map using the given formula
    return generic_filter(image, _local_contrast_window, size=size)

def compute_lc_stats(lc_map):
    mn = float(np.min(lc_map))
    mx = float(np.max(lc_map))
    mean = float(np.mean(lc_map))
    median = float(np.median(lc_map))
    std = float(np.std(lc_map))
    return {"min": mn, "max": mx, "mean": mean, "median": median, "std": std}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute local contrast map (8-connected).")
    parser.add_argument("image", nargs="?", default="resources/lion_b.png", help="Path to grayscale image")
    parser.add_argument("--size", type=int, default=3, help="Neighborhood size (odd integer)")
    parser.add_argument("--out", default=None, help="Output path for saved figure (if not interactive)")
    args = parser.parse_args()

    img = img_as_float(io.imread(args.image, as_gray=True))
    lc_map = compute_local_contrast(img, size=args.size)

    # Convert to grayscale units (0–255)
    lc_map_255 = lc_map * 255

    # Compute stats
    stats = compute_lc_stats(lc_map_255)
    print(f"Local contrast stats (0–255 scale): "
          f"min={stats['min']:.2f}, max={stats['max']:.2f}, mean={stats['mean']:.2f}, "
          f"median={stats['median']:.2f}, std={stats['std']:.2f}")

    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.imshow(img, cmap="gray")
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.imshow(lc_map, cmap="gray", vmin=0, vmax=np.max(lc_map))
    plt.title(f"Local contrast (size={args.size}x{args.size})")
    plt.axis("off")

    backend = mpl.get_backend()
    if backend in mpl.rcsetup.interactive_bk:
        plt.show()
    else:
        out_path = args.out or f"output_local_contrast_{args.size}.png"
        plt.savefig(out_path, bbox_inches='tight', dpi=150)
        print(f"Saved figure to {out_path} (backend: {backend})")
