from PIL import Image
import numpy as np

# === CONFIGURATION ===
input_path = "resources/lwy.png"
output_a_path = "output/lwy_dithered_109.png"
output_b_path = "output/lwy_dithered_5levels.png"

# --- (a) 1-bit threshold ---
T = 109

# --- (b) 5-level quantization ranges ---
def quantize_5_levels(val):
    if val < 20:
        return 0
    elif val < 40:
        return 64
    elif val < 60:
        return 128
    elif val < 120:
        return 192
    else:
        return 255

# === Jarvis–Judice–Ninke kernel ===
# Relative positions (dx, dy) and their weights
JJN_KERNEL = [
    (1, 0, 7), (2, 0, 5),
    (-2, 1, 3), (-1, 1, 5), (0, 1, 7), (1, 1, 5), (2, 1, 3),
    (-2, 2, 1), (-1, 2, 3), (0, 2, 5), (1, 2, 3), (2, 2, 1)
]
JJN_DIVISOR = 48.0


def apply_jjn_dither(img_array, quantize_fn):
    """Apply JJN dithering using a given quantization function."""
    h, w = img_array.shape
    pixels = img_array.astype(np.float32).copy()

    for y in range(h):
        for x in range(w):
            old_pixel = pixels[y, x]
            new_pixel = quantize_fn(old_pixel)
            pixels[y, x] = new_pixel
            error = old_pixel - new_pixel

            # Diffuse error according to JJN
            for dx, dy, weight in JJN_KERNEL:
                nx = x + dx
                ny = y + dy
                if 0 <= nx < w and 0 <= ny < h:
                    pixels[ny, nx] += error * (weight / JJN_DIVISOR)

    return np.clip(pixels, 0, 255).astype(np.uint8)


# === LOAD IMAGE ===
img = Image.open(input_path).convert("L")
gray = np.array(img, dtype=np.float32)

# --- (a) 1-bit reduction using threshold T = 109 ---
def quantize_1bit(val):
    return 255 if val >= T else 0

result_a = apply_jjn_dither(gray, quantize_1bit)
Image.fromarray(result_a, mode="L").save(output_a_path)
print(f"Saved 1-bit JJN dithered image (T={T}) → {output_a_path}")

# --- (b) 5-level quantization ---
result_b = apply_jjn_dither(gray, quantize_5_levels)
Image.fromarray(result_b, mode="L").save(output_b_path)
print(f"Saved 5-level JJN dithered image → {output_b_path}")
