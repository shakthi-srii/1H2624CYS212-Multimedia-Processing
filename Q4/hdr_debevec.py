import numpy as np
from gsolve import gsolve

def weight_function(z):
    z = np.asarray(z)
    return np.where(z <= 127, z, 255 - z)

def hdr_debevec(images, B, lambda_=50, num_px=150):
    images = [img.astype(np.float32) for img in images]
    num_images = len(images)
    H, W, _ = images[0].shape

    w = weight_function(np.arange(256))

    np.random.seed(0)
    idx = np.random.choice(H * W, num_px, replace=False)
    coords = np.array([(i // W, i % W) for i in idx])

    crf_channels = []
    log_irrad_channels = []

    for ch in range(3):
        Z = np.zeros((num_px, num_images), dtype=np.uint8)
        for j in range(num_images):
            img = images[j][:, :, ch]
            for i, (r, c) in enumerate(coords):
                Z[i, j] = img[r, c]

        g, lE = gsolve(Z, B, lambda_, w)
        crf_channels.append(g)
        log_irrad_channels.append(lE)

    return np.array(crf_channels), np.array(log_irrad_channels), w