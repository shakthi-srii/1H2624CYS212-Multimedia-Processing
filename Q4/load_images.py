import glob
import os
import numpy as np
import cv2

def load_images(image_dir, image_ext, root_dir):
    """
    Load exposure images for HDR.
    """
    path_pattern = os.path.join(root_dir, image_dir, "*" + image_ext)
    file_list = sorted(glob.glob(path_pattern))
    
    images = []
    
    for fpath in file_list:
        img = cv2.imread(fpath)
        if img is None:
            continue
        # Convert BGR (OpenCV default) to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)

    if len(images) == 0:
        raise ValueError(f"No images found in {path_pattern}")

    # Generate artificial exposure times (linear 1/n to 1.0)
    n = len(images)
    exposure_times = np.linspace(1/n, 1, n)
    B = np.log(exposure_times).astype(np.float32)

    return images, B