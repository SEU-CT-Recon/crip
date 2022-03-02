'''
    Utilities of crip.

    by z0gSh1u @ https://github.com/z0gSh1u/crip
'''


class CripException(BaseException):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


def cripAssert(cond, hint):
    if not cond:
        raise CripException(hint)