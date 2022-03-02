'''
    Postprocess module of crip.

    by z0gSh1u @ https://github.com/z0gSh1u/crip
'''
from .shared import *


def drawCircle(rec_img, r, center=None):
    """
        'Truncated Artifact Correction' specialized function.
        Draw circle before crop
    """
    theta = np.arange(0, 2 * np.pi, 0.01)
    if center is None:
        center = (rec_img.shape[0] // 2, rec_img.shape[1] // 2)

    x = center[0] + r * np.cos(theta)
    y = center[1] + r * np.sin(theta)

    return x, y


def fovCropRadiusReference(SOD, SDD, detectorWidth, pixelSpacing):
    """
        'Truncated Artifact Correction' specialized function.
        Reconstruction FOV section is independent of detector length

        SOD: Source object distance (mm)
        SDD: Source detector distance  (mm)
        detectorWidth: detector_elements * pixel_width (mm)
        pixelSpacing: pixel size of reconstructed image (mm)
    """
    # theta: scan angle width / 2
    # view bed plate as arc: arc = theta * r
    half_dw = detectorWidth / 2
    sin_theta = half_dw / np.sqrt(half_dw**2 + SDD**2)
    arc = SOD * np.arcsin(sin_theta)
    r_reference = arc / pixelSpacing

    # view bed plate as flat: flat = tan(theta) * r
    length = SOD / (SDD / detectorWidth) / 2
    r_reference2 = length / pixelSpacing

    # As we all know, tanx = x+x^3/3 + O(x^3), (|x| < pi/2),
    # so under no circumstances will r_reference greater than r_reference2
    up_flatten = SOD / (np.sqrt(half_dw**2 + SDD**2) / half_dw)
    r_reference3 = up_flatten / pixelSpacing

    return min(r_reference, r_reference2, r_reference3)  # r3 is the smallest


def cropCircleFOV(recon, radiusOrRatio, fillValue=0):
    '''
        Crop a circle FOV on `recon`.
    '''
    S, N, M = recon.shape
    if radiusOrRatio <= 1:
        radiusOrRatio = radiusOrRatio * min(N, M) / 2

    x = np.array([i for i in range(N)]) - N / 2 - 0.5
    y = np.array([i for i in range(M)]) - M / 2 - 0.5
    XX, YY = np.meshgrid(x, y)

    idx_zero = XX**2 + YY**2 > radiusOrRatio**2
    img_crop = recon
    img_crop[:, idx_zero] = fillValue

    return img_crop