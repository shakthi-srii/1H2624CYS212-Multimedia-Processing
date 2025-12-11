import cv2
import numpy as np

# ---------------------------------------------------
# 1. LOAD COLOR IMAGE
# ---------------------------------------------------
img = cv2.imread("Q5/image.jpg")

if img is None:
    raise FileNotFoundError("Image not found. Check file path!")

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
h, w, c = img.shape
print("Image size:", img.shape)

# ---------------------------------------------------
# 2. BOX FILTERS (5×5 & 20×20)
# ---------------------------------------------------
def box_filter(img, k, normalize=True):
    kernel = np.ones((k, k), np.float32)
    if normalize:
        kernel /= (k * k)
    return cv2.filter2D(img, -1, kernel)

# Save 5×5 results
box_5_norm = box_filter(img, 5, True)
box_5_un   = box_filter(img, 5, False)

cv2.imwrite("1_box_5_normalized.png",
            cv2.cvtColor(box_5_norm, cv2.COLOR_RGB2BGR))
cv2.imwrite("2_box_5_unnormalized.png",
            cv2.cvtColor(box_5_un, cv2.COLOR_RGB2BGR))

# Save 20×20 results
box_20_norm = box_filter(img, 20, True)
box_20_un   = box_filter(img, 20, False)

cv2.imwrite("3_box_20_normalized.png",
            cv2.cvtColor(box_20_norm, cv2.COLOR_RGB2BGR))
cv2.imwrite("4_box_20_unnormalized.png",
            cv2.cvtColor(box_20_un, cv2.COLOR_RGB2BGR))

print("BOX FILTERING DONE!")

# ---------------------------------------------------
# 3. SIGMA BASED ON FILTER SIZE 
# ---------------------------------------------------
# Standard Gaussian rule:  k = 6σ + 1  → σ = (k - 1) / 6

# You are computing Gaussian filter size from sigma,
# so we reverse the logic:
gauss_kernel_size = 21  # You can choose any odd size, e.g. 21
sigma = (gauss_kernel_size - 1) / 6

print("Sigma (based on kernel size) =", sigma)
print("Gaussian kernel size =", gauss_kernel_size)

# ---------------------------------------------------
# 4. 1D GAUSSIAN KERNEL (Separable)
# ---------------------------------------------------
def gaussian_1d(sigma, k):
    r = k // 2
    x = np.arange(-r, r + 1)
    g = np.exp(-(x * x) / (2 * sigma * sigma))
    return g.astype(np.float32)

G = gaussian_1d(sigma, gauss_kernel_size)
G_norm = G / np.sum(G)

# ---------------------------------------------------
# 5. SEPARABLE GAUSSIAN FILTERING
# ---------------------------------------------------
def separable_convolution(img, kernel):
    # Horizontal pass → Vertical pass
    temp = cv2.sepFilter2D(img, -1, kernel, np.array([1], np.float32))
    out  = cv2.sepFilter2D(temp, -1, np.array([1], np.float32), kernel)
    return out

# Gaussian (not normalized)
gauss_sep = separable_convolution(img, G)
cv2.imwrite("5_gaussian_separable.png",
            cv2.cvtColor(gauss_sep, cv2.COLOR_RGB2BGR))

# Gaussian (normalized)
gauss_sep_norm = separable_convolution(img, G_norm)
cv2.imwrite("6_gaussian_separable_normalized.png",
            cv2.cvtColor(gauss_sep_norm, cv2.COLOR_RGB2BGR))

print("GAUSSIAN FILTERING DONE!")
print("\nAll output images have been saved successfully!")