import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# ===== Parametry obrazu =====
size = 512              # rozmiar obrazu (piksele)
center = size // 2      # środek
r_max = center          # maksymalna odległość r w pikselach
I0 = 127                # amplituda jasności
const = 5120            # stała z zadania

# ===== Generowanie płytki Fresnela =====
y, x = np.indices((size, size))
r = np.sqrt((x - center)**2 + (y - center)**2)
I = I0 * np.cos(r**2 / const) + 128

out_path = "resources/PłytkaFresnela.png"
Image.fromarray(np.clip(I, 0, 255).astype(np.uint8)).save(out_path)

# TODO: Wyświetl obraz?