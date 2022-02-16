import cv2


def gaussianSmooth(projection, ksize, sigma):
    """
        Perform Gaussian smooth with kernel size = ksize and Gaussian \sigma = sigma (int or tuple (x, y)).
    """
    if isinstance(sigma, int):
        sigma = (sigma, sigma)
    return cv2.GaussianBlur(projection.astype(np.float32), ksize, sigmaX=sigma[0], sigmaY=sigma[1])
