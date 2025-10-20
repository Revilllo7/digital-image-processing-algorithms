from PIL import Image
import numpy as np

# === USER INPUT ===
input_path = "resources/lwy.png"   # your source file
output_path = "output/lwy_dithered.png"

T = float(input("Enter threshold T (0-255): ").strip() or 128)

# === LOAD IMAGE ===
img = Image.open(input_path).convert("L")  # convert to grayscale
pixels = np.array(img, dtype=np.float32)
h, w = pixels.shape

# === FLOYD–STEINBERG DITHERING ===
for y in range(h):
    for x in range(w):
        old_pixel = pixels[y, x]
        new_pixel = 255 if old_pixel >= T else 0  # threshold by T
        pixels[y, x] = new_pixel
        error = old_pixel - new_pixel

        # Distribute the error to neighbors (standard Floyd–Steinberg pattern)
        if x + 1 < w:
            pixels[y, x + 1] += error * 7 / 16
        if x - 1 >= 0 and y + 1 < h:
            pixels[y + 1, x - 1] += error * 3 / 16
        if y + 1 < h:
            pixels[y + 1, x] += error * 5 / 16
        if x + 1 < w and y + 1 < h:
            pixels[y + 1, x + 1] += error * 1 / 16

# === SAVE OUTPUT ===
pixels = np.clip(pixels, 0, 255).astype(np.uint8)
out = Image.fromarray(pixels, mode="L")
out.save(output_path)
print(f"Dithering complete. Saved to {output_path}")
