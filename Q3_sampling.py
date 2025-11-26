import cv2, numpy as np, matplotlib.pyplot as plt

# Read image in COLOR (BGR → RGB)
img = cv2.imread("image.png")
if img is None:
    raise ValueError("Image not found!")

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)

factors = [2, 4, 16]

def freq_sample(im, f):
    # FFT on each channel independently
    out = np.zeros_like(im)
    h, w, _ = im.shape

    for c in range(3):   # For R, G, B
        F = np.fft.fftshift(np.fft.fft2(im[:, :, c]))

        hk, wk = h // f, w // f
        mask = np.zeros((h, w))
        mask[h//2 - hk//2 : h//2 + hk//2,
             w//2 - wk//2 : w//2 + wk//2] = 1

        filtered = F * mask
        out[:, :, c] = np.abs(np.fft.ifft2(np.fft.ifftshift(filtered)))

    # Normalize to 0–255
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out

def spatial_sample(im, f):
    small = cv2.resize(im, None, fx=1/f, fy=1/f, interpolation=cv2.INTER_NEAREST)
    big = cv2.resize(small, (im.shape[1], im.shape[0]), interpolation=cv2.INTER_NEAREST)
    return big.astype(np.uint8)

plt.figure(figsize=(10, 8))
i = 1
for f in factors:
    plt.subplot(3, 2, i)
    plt.imshow(freq_sample(img, f))
    plt.title(f"Freq Sampling 1/{f}")
    plt.axis('off')
    i += 1

    plt.subplot(3, 2, i)
    plt.imshow(spatial_sample(img, f))
    plt.title(f"Spatial Sampling 1/{f}")
    plt.axis('off')
    i += 1

plt.tight_layout()
plt.show()
