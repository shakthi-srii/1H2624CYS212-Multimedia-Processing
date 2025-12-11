from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# -------- STEP 1: Read images --------
low = Image.open("low_light.jpg")
bright = Image.open("bright_light.jpg")

bright = bright.resize(low.size)

low_np = np.array(low)
bright_np = np.array(bright)

# Remove alpha channel if present
if low_np.shape[2] == 4:
    low_np = low_np[:, :, :3]
if bright_np.shape[2] == 4:
    bright_np = bright_np[:, :, :3]


# -------- FUNCTION: get 3 lowest bit planes --------
def lsb_3(img):
    b0 = (img >> 0) & 1
    b1 = (img >> 1) & 1
    b2 = (img >> 2) & 1
    return b0, b1, b2


# -------- STEP 2: Extract bit planes from BOTH images --------
low_b0, low_b1, low_b2 = lsb_3(low_np)
bright_b0, bright_b1, bright_b2 = lsb_3(bright_np)

# -------- STEP 3: Union of bit-planes (LOW + BRIGHT) --------
union_b0 = low_b0 | bright_b0
union_b1 = low_b1 | bright_b1
union_b2 = low_b2 | bright_b2

# -------- STEP 4: Reconstruct from UNION --------
reconstructed_union = (union_b0 * 1) + (union_b1 * 2) + (union_b2 * 4)
reconstructed_union = reconstructed_union.astype(np.uint8)

# -------- STEP 5: Difference from ORIGINAL (low-light original used) --------
difference = np.abs(low_np - reconstructed_union)

# -------- STEP 6: Display results --------
plt.figure()
plt.title("Low Light Original")
plt.imshow(low_np)
plt.axis("off")

plt.figure()
plt.title("Reconstructed Image (Union of 3 LSBs)")
plt.imshow(reconstructed_union)
plt.axis("off")

plt.figure()
plt.title("Difference Image")
plt.imshow(difference)
plt.axis("off")

plt.show()
