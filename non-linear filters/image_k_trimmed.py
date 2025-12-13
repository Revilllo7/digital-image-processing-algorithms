import numpy as np
from scipy.ndimage import generic_filter
import imageio.v2 as imageio

def k_trimmed_mean_filter(values, k=3):
    values = np.sort(values)
    return values[k:-k].mean()

# Wczytaj obraz, zapewnij 3 kanały (RGB)
img = imageio.imread("resources/input.png").astype(np.float32)

# Jeśli obraz jest w skali szarości (2D), powiel kanał do RGB
if img.ndim == 2:
    img = np.stack([img, img, img], axis=-1)

# Jeśli obraz ma kanał alfa (RGBA), odrzuć alfa
if img.ndim == 3 and img.shape[2] == 4:
    img = img[:, :, :3]

output = np.zeros_like(img)

# Przetwarzanie kanałów R, G, B osobno
for c in range(3):
    output[:, :, c] = generic_filter(
        img[:, :, c],
        function=k_trimmed_mean_filter,
        size=5,
        mode='reflect'
    )

# Zapis
output = np.clip(output, 0, 255).astype(np.uint8)
imageio.imwrite("output/output_ktrimmed.png", output)