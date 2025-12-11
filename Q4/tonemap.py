import numpy as np
import cv2
import matplotlib.pyplot as mp_plt

def plot_and_save(img, img_name, img_title):
    params = [cv2.IMWRITE_JPEG_QUALITY, 90]
    
    mp_plt.figure(figsize=(10, 10))
    mp_plt.imshow(img)
    mp_plt.title(img_title)
    mp_plt.axis('off')

    # Convert to 0-255 uint8
    img_save = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    
    # Convert RGB to BGR for OpenCV saving
    cv2.imwrite(img_name + ".jpg", img_save[:, :, ::-1], params)

def reinhard_tonemap(E, gamma=1/2.2, alpha=0.18):
    E = E.astype(np.float32)
    
    # Luminance
    L = 0.2126 * E[:, :, 0] + 0.7152 * E[:, :, 1] + 0.0722 * E[:, :, 2]
    L_safe = np.clip(L, 1e-8, None)
    
    # Log-average
    L_avg = np.exp(np.mean(np.log(L_safe)))
    
    # Scale
    L_scaled = (alpha / L_avg) * L_safe
    
    # Tone map
    L_tone = L_scaled / (1 + L_scaled)
    
    # Reconstruct Color
    tonemapped = np.zeros_like(E)
    ratio = L_tone / L_safe
    for ch in range(3):
        tonemapped[:, :, ch] = E[:, :, ch] * ratio
        
    # Gamma Correction
    return np.power(np.clip(tonemapped, 0, 1), gamma)