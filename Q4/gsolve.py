import numpy as np

def gsolve(Z, B, lambda_, w, Zmin=0, Zmax=255):
    """
    Solve for CRF using Debevec & Malik method.
    """
    n = Zmax - Zmin + 1
    num_px, num_im = Z.shape

    A = np.zeros((num_px * num_im + n, n + num_px), dtype=np.float64)
    b = np.zeros(A.shape[0], dtype=np.float64)

    k = 0

    # 1. Data-fitting
    for i in range(num_px):
        for j in range(num_im):
            z_ij = Z[i, j]
            weight = w[z_ij]
            A[k, z_ij] = weight
            A[k, n + i] = -weight
            b[k] = weight * B[j]
            k += 1

    # 2. Anchor
    mid = (Zmin + Zmax) // 2
    A[k, mid] = 1.0
    k += 1

    # 3. Smoothness
    for z in range(Zmin + 1, Zmax - 1):
        weight = w[z]
        A[k, z - 1] = lambda_ * weight
        A[k, z]     = -2 * lambda_ * weight
        A[k, z + 1] = lambda_ * weight
        k += 1

    # 4. Solve
    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    g = x[:n]
    lE = x[n:]

    return g, lE