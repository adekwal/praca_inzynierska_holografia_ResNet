import numpy as np

def calc_RMSE(image1, image2):
    if image1.shape != image2.shape:
        raise ValueError("Different sizes")

    mse = np.mean((image1 - image2) ** 2)
    return np.round(np.sqrt(mse), 3)

def calc_SSIM(image1, image2):
    K1, K2 = 0.01, 0.03
    L = np.max([np.max(image1), np.max(image2)]) - np.min([np.min(image1), np.min(image2)])
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2

    mean1 = np.mean(image1)
    mean2 = np.mean(image2)
    var1 = np.var(image1)
    var2 = np.var(image2)
    cov12 = np.mean((image1 - mean1) * (image2 - mean2))

    numerator = (2 * mean1 * mean2 + C1) * (2 * cov12 + C2)
    denominator = (mean1**2 + mean2**2 + C1) * (var1 + var2 + C2)
    ssim = numerator / denominator

    return np.round(ssim, 3)