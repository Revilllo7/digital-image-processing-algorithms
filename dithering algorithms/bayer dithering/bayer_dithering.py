import numpy as np
from PIL import Image

def bayer_4x4_dithering():
    # Load image and convert to grayscale
    img = Image.open("resources/lwy.png").convert("L")
    img_arr = np.array(img, dtype=np.float32)
    
    # Define the 4×4 Bayer matrix
    bayer = np.array([
        [0,  8,  2, 10],
        [12, 4, 14, 6],
        [3, 11, 1, 9],
        [15, 7, 13, 5]
    ], dtype=np.float32)
    
    # Normalize Bayer matrix to 0–1 thresholds
    threshold = (bayer + 0.5) / 16.0
    
    # 5 gray levels
    gray_levels = np.array([0, 64, 128, 192, 255], dtype=np.float32)
    n_levels = len(gray_levels)
    
    # Determine the quantization step
    step = 255.0 / (n_levels - 1)
    
    # Prepare output image
    out = np.zeros_like(img_arr)
    
    h, w = img_arr.shape
    for y in range(h):
        for x in range(w):
            # Normalize pixel intensity to 0–1
            norm = img_arr[y, x] / 255.0
            
            # Corresponding threshold from Bayer matrix (tile repeating)
            t = threshold[y % 4, x % 4]
            
            # Apply variable threshold and scale back to 0–255
            val = np.floor((norm + t / (n_levels - 1)) * (n_levels - 1))
            val = np.clip(val, 0, n_levels - 1)
            
            out[y, x] = gray_levels[int(val)]
    
    # Save the output image
    Image.fromarray(out.astype(np.uint8)).save("output/lwy_dithered_5levels.png")
    print(f"Saved dithered image to output/lwy_dithered_5levels.png")


bayer_4x4_dithering()
