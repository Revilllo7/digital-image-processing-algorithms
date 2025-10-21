from PIL import Image
import numpy as np

# Input/output paths
input_path = "potworek_pixelart.png"
output_path = "potworek_pixelart_scaled_D.png"

# Target size
target_width, target_height = 500, 650

def custom_interpolate(channel, new_w, new_h):
    h_in, w_in = channel.shape
    result = np.zeros((new_h, new_w), dtype=np.float32)
    scale_x = w_in / new_w
    scale_y = h_in / new_h

    for y_out in range(new_h):
        for x_out in range(new_w):
            src_x = x_out * scale_x
            src_y = y_out * scale_y

            x0 = int(np.floor(src_x))
            y0 = int(np.floor(src_y))
            x1 = min(x0 + 1, w_in - 1)
            y1 = min(y0 + 1, h_in - 1)

            neighbors = np.array([
                channel[y0, x0],
                channel[y0, x1],
                channel[y1, x0],
                channel[y1, x1],
            ], dtype=np.float32)

            val = (neighbors.max() + neighbors.min()) / 2.0
            result[y_out, x_out] = np.clip(val, 0, 255)

    return result.astype(np.uint8)

# Load image and split channels
img = Image.open(input_path).convert("RGB")
r, g, b = img.split()

# Convert to NumPy arrays
r = np.array(r)
g = np.array(g)
b = np.array(b)

# Scale each channel using custom interpolation
r_scaled = custom_interpolate(r, target_width, target_height)
g_scaled = custom_interpolate(g, target_width, target_height)
b_scaled = custom_interpolate(b, target_width, target_height)

# Merge and save
out_img = Image.merge(
    "RGB",
    [Image.fromarray(r_scaled), Image.fromarray(g_scaled), Image.fromarray(b_scaled)]
)
out_img.save(output_path)

print(f"Saved scaled image to {output_path}")
