import numpy as np

def compute_irradiance(g_channels, w, images, B):
    """
    Reconstruct HDR radiance map from image stack.
    """
    num_images = len(images)
    H, W, num_channels = images[0].shape
    
    # Vectorized lookup
    Z = np.stack(images, axis=2).astype(np.uint8) 
    log_irradiance_map = np.zeros((H, W, num_channels), dtype=np.float32)

    for ch in range(num_channels):
        g = g_channels[ch]
        Z_ch = Z[:, :, :, ch]
        
        g_lookup = g[Z_ch]
        w_lookup = w[Z_ch]
        
        # Broadcast B to match shape
        B_broad = B[np.newaxis, np.newaxis, :]
        
        numerator = np.sum(w_lookup * (g_lookup - B_broad), axis=2)
        denominator = np.sum(w_lookup, axis=2)
        
        denominator = np.where(denominator == 0, 1e-8, denominator)
        log_irradiance_map[:, :, ch] = numerator / denominator

    return np.exp(log_irradiance_map)