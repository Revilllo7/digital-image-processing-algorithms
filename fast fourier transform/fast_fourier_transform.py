# Source - https://stackoverflow.com/questions/38476359/fft-on-image-with-python
# Posted by Ahmed Fasih, modified by community. See post 'Timeline' for change history
# Retrieved 2025-11-09, License - CC BY-SA 3.0

# It doesn't work correctly btw

from PIL import Image
import numpy as np
import numpy.fft as fp

## Functions to go from image to frequency-image and back
def im2freq(data):
    # compute 2D FFT over the two spatial axes (0,1); supports multi-channel (last axis)
    # use full FFT (fft2) so the frequency-image has the same spatial resolution as the input
    return fp.fft2(data, axes=(0, 1))

def freq2im(f):
    # compute inverse 2D FFT and return the real-valued image with the original spatial shape
    # convert to float32 to match the input dtype
    im = fp.ifft2(f, axes=(0, 1)).real
    return im.astype(np.float32)

## Helper functions to rescale a frequency-image to [0, 255] and save
def touint8(x):
    # use magnitude of complex frequency data, subtract min per-channel, then normalize
    mag = np.abs(x)
    mag = mag - np.amin(mag, axis=(0,1), keepdims=True)
    maxval = np.amax(mag, axis=(0,1), keepdims=True)
    # avoid division by zero
    maxval = np.maximum(maxval, 1e-12)
    out = (mag / maxval) * (256 - 1e-4)
    return out.astype(np.uint8)

def arr2im(data, fname):
    # Accept 2D (H,W) or 3D (H,W,C) uint8 arrays and create the appropriate PIL image.
    data = np.asarray(data, dtype=np.uint8)
    if data.ndim == 2:
        img = Image.fromarray(data, mode='L')
    elif data.ndim == 3:
        c = data.shape[2]
        if c == 3:
            img = Image.fromarray(data, mode='RGB')
        elif c == 4:
            img = Image.fromarray(data, mode='RGBA')
        elif c < 3:
            # pad to 3 channels
            pad = np.zeros((data.shape[0], data.shape[1], 3), dtype=data.dtype)
            pad[..., :c] = data
            img = Image.fromarray(pad, mode='RGB')
        else:
            # more than 4 channels: take first 3 channels
            img = Image.fromarray(data[..., :3], mode='RGB')
    else:
        raise ValueError(f"Unsupported data shape for arr2im: {data.shape}")
    img.save(fname)

# Read in data file and transform when run as a script
if __name__ == "__main__":
    data = np.array(Image.open('test.png')).astype(np.float32) / 255.0

    freq = im2freq(data)
    back = freq2im(freq)

    # Make sure the forward and backward transforms work within floating-point tolerance
    # cast the reconstructed array to the input dtype and allow small numerical differences
    assert np.allclose(data, back.astype(data.dtype), rtol=1e-5, atol=1e-6)

    arr2im(touint8(freq), 'freq.png')
