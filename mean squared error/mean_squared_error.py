import numpy as np
from skimage import io, img_as_float
from skimage.color import rgba2rgb, gray2rgb
from pathlib import Path

# make resource paths relative to this script
BASE = Path(__file__).parent / "resources"

# === Load images ===
ref = img_as_float(io.imread(str(BASE / "osaRGB_png.png")))   # reference image
gif = img_as_float(io.imread(str(BASE / "osaRGB_gif.gif")))
jpg = img_as_float(io.imread(str(BASE / "osaRGB_jpg.jpg")))

def ensure_rgb(img):
    # take first frame if image is a frame stack like (N, H, W, C) or (1, H, W, C)
    if img.ndim == 4:
        img = img[0]
    # grayscale -> RGB
    if img.ndim == 2:
        img = gray2rgb(img)
    # RGBA -> RGB
    if img.ndim == 3 and img.shape[2] == 4:
        img = rgba2rgb(img)
    return img

ref = ensure_rgb(ref)
gif = ensure_rgb(gif)
jpg = ensure_rgb(jpg)

# === Check sizes ===
if ref.shape != gif.shape:
    raise ValueError(f"GIF image has different dimensions than reference PNG: {gif.shape} vs {ref.shape}")
if ref.shape != jpg.shape:
    raise ValueError(f"JPEG image has different dimensions than reference PNG: {jpg.shape} vs {ref.shape}")


# === Define MSE function for RGB images ===
def mse_rgb(img1, img2):
    
    # Computes Mean Squared Error (MSE) between two RGB images.
    # MSE is calculated per-channel and averaged across channels.
    diff = img1 - img2
    mse_channels = np.mean(diff ** 2, axis=(0, 1))   # mean for each channel (R, G, B)
    mse_total = np.mean(mse_channels)                # average across channels
    return mse_total, mse_channels

# === Compute MSE ===
mse_gif, mse_gif_channels = mse_rgb(ref, gif)
mse_jpg, mse_jpg_channels = mse_rgb(ref, jpg)

# === Print results ===
print("MSE results (relative to reference PNG):")
print(f"GIF  - per channel: R={mse_gif_channels[0]:.6f}, G={mse_gif_channels[1]:.6f}, B={mse_gif_channels[2]:.6f}")
print(f"      average MSE: {mse_gif:.6f}")
print(f"JPEG - per channel: R={mse_jpg_channels[0]:.6f}, G={mse_jpg_channels[1]:.6f}, B={mse_jpg_channels[2]:.6f}")
print(f"      average MSE: {mse_jpg:.6f}")

# === Determine which is closer to reference ===
better = "GIF" if mse_gif < mse_jpg else "JPEG"
print(f"\n => The image with better quality (lower MSE) is: {better}")

# === Scale results by 255^2 for 8-bit images ===
scale_factor = 255.0 ** 2
mse_gif_scaled = mse_gif * scale_factor
mse_jpg_scaled = mse_jpg * scale_factor
print("\nMSE results scaled to 8-bit range (0-255):")
print(f"GIF  - average MSE (scaled): {mse_gif_scaled:.2f}")
print(f"JPEG - average MSE (scaled): {mse_jpg_scaled:.2f}")
better_scaled = "GIF" if mse_gif_scaled < mse_jpg_scaled else "JPEG"
print(f"\n => The image with better quality (lower MSE) in 8-bit scale is: {better_scaled}")
