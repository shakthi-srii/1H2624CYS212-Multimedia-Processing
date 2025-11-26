# kmeans_quant.py
import os
import numpy as np
import cv2
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from math import log2

def psnr(mse, max_pixel=255.0):
    if mse == 0:
        return float('inf')
    return 10 * np.log10((max_pixel**2) / mse)

def quantize_kmeans(img, K, sample_fraction=1.0, random_state=0, max_iter=100):
    orig_shape = img.shape
    is_color = (img.ndim == 3 and img.shape[2] == 3)
    X = img.reshape(-1, img.shape[-1] if is_color else 1).astype(np.float32)

    if sample_fraction < 1.0:
        n = X.shape[0]
        idx = np.random.RandomState(random_state).choice(n, int(n*sample_fraction), replace=False)
        X_sample = X[idx]
    else:
        X_sample = X

    kmeans = KMeans(n_clusters=K, random_state=random_state, n_init=10, max_iter=max_iter)
    kmeans.fit(X_sample)

    labels = kmeans.predict(X)
    centers = np.clip(kmeans.cluster_centers_, 0, 255).astype(np.uint8)

    Xq = centers[labels]
    quantized = Xq.reshape(orig_shape)

    diff = (quantized.astype(np.float32) - img.astype(np.float32)) ** 2
    mse = diff.mean()
    return quantized, mse, centers

def run_on_image(image_path, Ks=[2,4,8,16,32,64], out_dir='results', sample_fraction=1.0):
    os.makedirs(out_dir, exist_ok=True)
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot read {image_path}")

    img_display = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = []
    for K in Ks:
        quantized, mse, centers = quantize_kmeans(img_display, K, sample_fraction=sample_fraction)
        r = log2(K)
        p = psnr(mse)

        outname = os.path.join(out_dir, f"quant_K{K}.png")
        save_img = cv2.cvtColor(quantized, cv2.COLOR_RGB2BGR)
        cv2.imwrite(outname, save_img)

        results.append({'K': K, 'Rate_bits_per_pixel': r,
                        'MSE': mse, 'PSNR_dB': p, 'out_image': outname})

        print(f"K={K:3d} | Rate={r:.2f} bpp | MSE={mse:.2f} | PSNR={p:.2f} dB")

    rates = [row['Rate_bits_per_pixel'] for row in results]
    psnrs = [row['PSNR_dB'] for row in results]

    plt.figure(figsize=(6,4))
    plt.plot(rates, psnrs, marker='o')
    plt.xlabel('Rate (bits/pixel = log2(K))')
    plt.ylabel('PSNR (dB)')
    plt.title('Rate-Distortion: K-means Quantization')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'rate_distortion.png'))
    plt.close()

if __name__ == "__main__":
    run_on_image("umbrella_image.png", Ks=[2,4,8,16,32,64], out_dir="results")
