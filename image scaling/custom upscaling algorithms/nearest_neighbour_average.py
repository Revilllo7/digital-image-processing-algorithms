from PIL import Image
import numpy as np

input_path = "resources/potworek_pixelart.png"
output_path = "output/potworek_pixelart_scaled_B.png"

target_width, target_height = 500, 650

def two_nearest_average(channel, new_w, new_h):
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

            dx = src_x - x0
            dy = src_y - y0

            coords = [(x0, y0), (x1, y0), (x0, y1), (x1, y1)]
            dists = [
                np.sqrt((src_x - x)**2 + (src_y - y)**2)
                for (x, y) in coords
            ]
            vals = [
                float(channel[y, x])
                for (x, y) in coords
            ]

            # Find indices of two nearest pixels
            idx_sorted = np.argsort(dists)
            v1 = vals[idx_sorted[0]]
            v2 = vals[idx_sorted[1]]

            result[y_out, x_out] = (v1 + v2) / 2.0

    return np.clip(result, 0, 255).astype(np.uint8)

# Load and split
img = Image.open(input_path).convert("RGB")
r, g, b = img.split()

r_scaled = two_nearest_average(np.array(r), target_width, target_height)
g_scaled = two_nearest_average(np.array(g), target_width, target_height)
b_scaled = two_nearest_average(np.array(b), target_width, target_height)

out = Image.merge("RGB", [
    Image.fromarray(r_scaled),
    Image.fromarray(g_scaled),
    Image.fromarray(b_scaled)
])
out.save(output_path)
print(f"Saved scaled image to {output_path}")
