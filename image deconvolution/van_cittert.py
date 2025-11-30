# Python code to perform Van-Cittert deconvolution on the provided image file.
# It reads the image at the path provided by the developer message, applies Van-Cittert for k=2,5,15,
# saves reconstructions and absolute differences, and displays the three reconstructions and diffs inline.

from pathlib import Path
import numpy as np
from PIL import Image
from scipy.signal import fftconvolve
import imageio
import matplotlib.pyplot as plt

input_path = Path("resources/orig.png")

if not input_path.exists():
    raise FileNotFoundError(f"Input image not found at {input_path}. Please upload or place the file there.")

# Read image and convert to grayscale float32
img = Image.open(input_path).convert("L")  # convert to grayscale
g = np.asarray(img, dtype=np.float32)

# Normalize image to original dynamic range preserved (no scaling to 0-1 to keep values)
# But we will save as 32-bit TIFF to preserve floats (PIL doesn't save float TIFF directly well,
# so we scale to 0-65535 and save uint16, and also save PNG previews)
def save_float_as_uint16(path, arr):
    # Clip to finite, then scale to 0-65535 for uint16 saving
    arr = np.array(arr, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    mn, mx = arr.min(), arr.max()
    if mx == mn:
        scaled = np.zeros_like(arr, dtype=np.uint16)
    else:
        scaled = ((arr - mn) / (mx - mn) * 65535.0).astype(np.uint16)
    imageio.imwrite(str(path), scaled)
    return mn, mx

out_dir = Path("output")
out_dir.mkdir(exist_ok=True)

# Define PSF kernel (5x5) and normalize by 256
kernel_raw = np.array([
    [1, 4, 6, 4, 1],
    [4,16,24,16,4],
    [6,24,36,24,6],
    [4,16,24,16,4],
    [1,4,6,4,1]
], dtype=np.float32)
psf = kernel_raw / 256.0  # normalized

# Van-Cittert deconvolution function
def van_cittert(g, psf, alpha=1.0, iterations=10, boundary='symm'):
    """
    Van Cittert deconvolution: f_{n+1} = f_n + alpha * (g - h * f_n)
    Uses FFT convolution for speed (fftconvolve) with 'same' mode.
    """
    f = g.copy().astype(np.float32)
    for i in range(iterations):
        conv = fftconvolve(f, psf, mode='same')
        f = f + alpha * (g - conv)
    return f

# Run for k = 2,5,15 and save results + absolute differences
ks = [2,5,15]
saved_files = []
for k in ks:
    recon = van_cittert(g, psf, alpha=1.0, iterations=k)
    recon_name = out_dir / f"recon_k{k}.tif"
    # Save reconstruction as uint16 TIFF (scaled), but also save PNG preview
    mn, mx = save_float_as_uint16(recon_name, recon)
    # Save PNG preview (8-bit) for quick viewing scaled to 0-255
    recon_preview = ((recon - recon.min()) / (recon.max() - recon.min()) * 255).astype(np.uint8) if recon.max()!=recon.min() else np.zeros_like(recon, dtype=np.uint8)
    imageio.imwrite(str(out_dir / f"recon_k{k}_preview.png"), recon_preview)
    
    # Absolute difference |g - recon|
    diff = np.abs(g - recon)
    diff_name = out_dir / f"absdiff_k{k}.tif"
    save_float_as_uint16(diff_name, diff)
    diff_preview = ((diff - diff.min()) / (diff.max() - diff.min()) * 255).astype(np.uint8) if diff.max()!=diff.min() else np.zeros_like(diff, dtype=np.uint8)
    imageio.imwrite(str(out_dir / f"absdiff_k{k}_preview.png"), diff_preview)
    
    saved_files.append((recon_name, out_dir / f"recon_k{k}_preview.png", diff_name, out_dir / f"absdiff_k{k}_preview.png"))

# Display the three reconstructions and diffs using matplotlib (one plot per image as required)
fig, axes = plt.subplots(3,2, figsize=(10,15))
for idx, k in enumerate(ks):
    recon_preview = imageio.imread(str(out_dir / f"recon_k{k}_preview.png"))
    diff_preview = imageio.imread(str(out_dir / f"absdiff_k{k}_preview.png"))
    axes[idx,0].imshow(recon_preview, cmap='gray')
    axes[idx,0].set_title(f"Reconstruction k={k}")
    axes[idx,0].axis('off')
    axes[idx,1].imshow(diff_preview, cmap='gray')
    axes[idx,1].set_title(f"|orig - recon| k={k}")
    axes[idx,1].axis('off')

plt.tight_layout()
plt.show()

# Print saved file locations
print("Saved files (TIFF reconstructions and diffs are scaled to uint16):")
for recon_tiff, recon_png, diff_tiff, diff_png in saved_files:
    print(f"- Reconstruction: {recon_tiff}  (preview: {recon_png})")
    print(f"  Absolute diff:  {diff_tiff}  (preview: {diff_png})")

# Provide a small function to compute MSE and PSNR between original and reconstructions
def mse(a,b):
    return np.mean((a.astype(np.float64) - b.astype(np.float64))**2)

print("\nQuality metrics (MSE):")
for k in ks:
    recon = imageio.imread(str(out_dir / f"recon_k{k}_preview.png")).astype(np.float32)
    # convert preview back to approximate original scale for metric (not exact)
    m = mse(g, (recon/255.0)*(g.max()-g.min()) + g.min())
    print(f" k={k}: approx MSE = {m:.3f}")

# Show download path
print(f"\nAll results are in: {out_dir}")