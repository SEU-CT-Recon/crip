'''
    Postprocess module of crip.

    by z0gSh1u @ https://github.com/z0gSh1u/crip
'''
from .shared import *
from matplotlib import pyplot as plt


def drawCircle(rec_img, r, center=None, style='b-.'):
    """
        'Truncated Artifact Correction' specialized function.
        Draw circle before crop
    """
    theta = np.arange(0, 2 * np.pi, 0.01)
    if center is None:
        center = (rec_img.shape[0]//2, rec_img.shape[1]//2)

    x = center[0] + r * np.cos(theta)
    y = center[1] + r * np.sin(theta)

    plt.plot(x, y, style)


def fovCropRadiusReference(SOD, SDD, detector_width, rec_pixel_width):
    """
        'Truncated Artifact Correction' specialized function.
        Reconstruction FOV section is independent of detector length

        SOD: Source objective distance (mm)
        SDD: Source detector distance  (mm)
        detector_width: detector_elements * pixel_width (mm)
        rec_pixel_width: reconstructed image length / pixel number
    """
    # theta: scan angle width / 2
    # view bed plate as arc: arc = theta * r
    half_dw = detector_width / 2
    sin_theta = half_dw / np.sqrt(half_dw**2 + SDD**2)
    arc = SOD * np.arcsin(sin_theta)
    r_reference = arc / rec_pixel_width

    # view bed plate as flat: flat = tan(theta) * r
    length = SOD / (SDD / detector_width) / 2
    r_reference2 = length / rec_pixel_width

    """
        As we all know, tanx = x+x^3/3 + O(x^3), (|x| < pi/2),
        so under no circumstances will r_reference greater than r_reference2
    """

    up_flatten = SOD / (np.sqrt(half_dw**2 + SDD**2) / half_dw)
    r_reference3 = up_flatten / rec_pixel_width

    return min(r_reference, r_reference2, r_reference3)  # r3 is the smallest


def cropCircleFOV(rec_img, radius, fillValue=0):
    S, N, M = rec_img.shape
    if radius <= 1:
        radius = radius * min(N, M) / 2

    x = np.array([i for i in range(N)]) - N / 2 - 0.5
    y = np.array([i for i in range(M)]) - M / 2 - 0.5
    XX, YY = np.meshgrid(x, y)

    idx_zero = XX ** 2 + YY ** 2 > radius ** 2
    img_crop = rec_img
    img_crop[:, idx_zero] = fillValue

    return img_crop