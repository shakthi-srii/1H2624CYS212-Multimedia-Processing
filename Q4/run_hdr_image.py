import numpy as np
import os, sys
import cv2
import matplotlib.pyplot as mp_plt

from load_images import load_images
from hdr_debevec import hdr_debevec
from compute_irradiance import compute_irradiance
from tonemap import reinhard_tonemap, plot_and_save

def run_hdr(root_dir, image_dir, image_ext, COMPUTE_CRF, kwargs):
    lambda_ = kwargs['lambda_']
    num_px  = kwargs['num_px']
    gamma   = kwargs['gamma']
    alpha   = kwargs['alpha']

    # 1. Load Images
    print("Loading images...")
    images, B = load_images(image_dir, image_ext, root_dir)

    # --- SAVE SAMPLE EXPOSURES PLOT ---
    if len(images) >= 2:
        plot_idx = np.random.choice(len(images), 2, replace=False)
        mp_plt.figure(figsize=(16, 8))
        mp_plt.subplot(1, 2, 1)
        mp_plt.imshow(images[plot_idx[0]])
        mp_plt.title(f"Exposure: {np.exp(B[plot_idx[0]]):.5f} sec")
        mp_plt.subplot(1, 2, 2)
        mp_plt.imshow(images[plot_idx[1]])
        mp_plt.title(f"Exposure: {np.exp(B[plot_idx[1]]):.5f} sec")
        sample_path = os.path.join(root_dir, "sample_exposures.png")
        mp_plt.savefig(sample_path)
        print(f"Sample exposures saved to: {sample_path}")

    # 2. CRF (Debevec)
    crf_file = os.path.join(root_dir, "crf.npz")
    if isinstance(COMPUTE_CRF, str):
        COMPUTE_CRF = COMPUTE_CRF.lower() in ["true", "1", "yes"]

    if COMPUTE_CRF:
        print("Computing CRF...")
        crf, lE, w = hdr_debevec(images, B, lambda_, num_px)
        np.savez(crf_file, crf=crf, log_irrad=lE, w=w)
    else:
        print("Loading CRF from file...")
        data = np.load(crf_file)
        crf, lE, w = data["crf"], data["log_irrad"], data["w"]

    # --- SAVE CRF PLOT ---
    mp_plt.figure(figsize=(10, 6))
    colors = ['r', 'g', 'b']
    labels = ['Red Channel', 'Green Channel', 'Blue Channel']
    for ch in range(3):
        mp_plt.plot(crf[ch], color=colors[ch], label=labels[ch])
    mp_plt.title('Camera Response Function (CRF)')
    mp_plt.xlabel('Pixel Value (Z)')
    mp_plt.ylabel('Log Exposure (g(Z))')
    mp_plt.legend()
    mp_plt.grid(True)
    crf_plot_path = os.path.join(root_dir, "crf_plot.png")
    mp_plt.savefig(crf_plot_path)
    print(f"CRF plot saved to: {crf_plot_path}")

    # 3. Compute Radiance Map
    print("Computing Irradiance Map...")
    irradiance_map = compute_irradiance(crf, w, images, B)

    # 4. Tone Mapping
    print("Tone Mapping...")
    tonemapped_img = reinhard_tonemap(irradiance_map, gamma, alpha)

    # --- FIX FOR NEGATIVE IMAGE ---
    # Invert the image back to normal
    tonemapped_img = 1.0 - tonemapped_img
    # ------------------------------

    # 5. Save Output
    out_dir = os.path.join(root_dir, image_dir)
    print(f"Saving HDR result to {out_dir}.jpg...")
    plot_and_save(tonemapped_img, out_dir, "Globally Tonemapped Image")

    return tonemapped_img

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: python run_hdr_image.py <ROOT> <IMG_DIR> <EXT> <COMPUTE_CRF>")
        sys.exit(1)

    ROOT = sys.argv[1]
    IMG_DIR = sys.argv[2]
    EXT = sys.argv[3]
    CRF = sys.argv[4]

    kwargs = {'lambda_': 50, 'num_px': 150, 'gamma': 1/2.2, 'alpha': 0.4}
    
    run_hdr(ROOT, IMG_DIR, EXT, CRF, kwargs)