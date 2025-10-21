import numpy as np
from PIL import Image

# === Load grayscale image ===
img = Image.open("resources/lwy.png").convert("L")
arr = np.array(img, dtype=np.float32)

# === 3x3 Dithering matrix ===
D = np.array([
    [7, 1, 5],
    [3, 0, 2],
    [4, 8, 6]
], dtype=np.float32)

N = 3
M = N * N  # 9

# === Apply ordered dithering ===
height, width = arr.shape
result = np.zeros_like(arr)

for y in range(height):
    for x in range(width):
        # Find threshold from D
        threshold = ((D[y % N, x % N] + 0.5) / M) * 255
        result[y, x] = 255 if arr[y, x] > threshold else 0

# === Save result ===
Image.fromarray(result.astype(np.uint8)).save("output/lwy_dithered.png")
print("Saved 'output/lwy_dithered.png'")