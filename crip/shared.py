import numpy as np
import cv2


def rotate(img, deg):
    """
        Rotate img
    """
    deg = int(deg % 360)
    if deg == 0:
        return img
    degToCode = {
        '90': cv2.ROTATE_90_CLOCKWISE,
        '180': cv2.ROTATE_180,
        '270': cv2.ROTATE_90_COUNTERCLOCKWISE,
    }
    return cv2.rotate(img, degToCode[str(deg)])


def resize(projection, dsize=None, fx=None, fy=None, interp='bicubic'):
    """
        Resize the projection to `dsize` (if dsize is not None) or scale by ratio `(fx, fy)`
        using `interp` (bicubic, linear, nearest available).
    """
    if dsize is None:
        assert fx is not None or fy is not None
        fx = fx if fx else 1
        fy = fy if fy else 1
    else:
        assert fx is None and fy is None
    interp_ = {'bicubic': cv2.INTER_CUBIC, 'linear': cv2.INTER_LINEAR, 'nearest': cv2.INTER_NEAREST}
    return cv2.resize(projection.astype(np.float32), dsize, None, fx, fy, interpolation=interp_[interp])


def binning(projection, binning=(1, 1)):
    """
        Perform binning on row and col directions. `binning=(rowBinning, colBinning)`.
    """
    res = np.array(projection[::binning[0], ::binning[1]])
    return res


def splitToSingle():
    pass


def mergeToStack():
    pass