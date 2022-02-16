import os


def getExtName(filename: str):
    basename = os.path.basename(filename)
    split = basename.split('.')
    if len(split) == 1:
        return None
    return split[-1]