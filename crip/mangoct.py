'''
    MangoCT integration of crip. See
    https://gitee.com/njjixu/mangoct
    https://github.com/z0gSh1u/mangoct
    https://github.com/CandleHouse/mandoct

    https://github.com/z0gSh1u/crip
'''

__all__ = ['Mgfbp', 'Mgfpj']

import os

from .utils import sysPlatform
from ._typing import *


# TODO
class MgfbpConfig:
    def __init__(self):
        raise 'Not implemented.'


# TODO
class MgfpjConfig:
    def __init__(self):
        raise 'Not implemented.'


class Mgfbp:
    def __init__(self, exe: str = 'mgfbp', cudaDevice=0):
        '''
            Initialize a handler object to use the FBP tool in mangoct.
            `exe` is the path to the executable.
        '''
        self.exe = exe
        self.cudaDevice = cudaDevice
        self.cmd = []
        self._buildCmd()

    def _buildCmd(self):
        platform = sysPlatform()

        if platform == 'Windows':
            self.cmd.append(f'set CUDA_VISIBLE_DEVICES={self.cudaDevice}')
        elif platform == 'Linux':
            self.cmd.append(f'export CUDA_VISIBLE_DEVICES={self.cudaDevice}')

        self.cmd.append(f'"{self.exe}" "<1>"')

    def exec(self, confPath: str):
        for cmd in self.cmd:
            cmd = cmd.replace('<1>', confPath)
            os.system(cmd)


class Mgfpj:
    def __init__(self, exe: str = 'mgfpj', cudaDevice=0) -> None:
        '''
            Initialize a handler object to use the FPJ tool in mangoct.
            `exe` is the path to the executable.
        '''
        self.exe = exe
        self.cudaDevice = cudaDevice
        self.cmd = None
        self._buildCmd()

    def _buildCmd(self):
        platform = sysPlatform()

        if platform == 'Windows':
            self.cmd.append(f'set CUDA_VISIBLE_DEVICES={self.cudaDevice}')
        elif platform == 'Linux':
            self.cmd.append(f'export CUDA_VISIBLE_DEVICES={self.cudaDevice}')

        self.cmd.append(f'"{self.exe}" "<1>"')

    def exec(self, confPath: str):
        for cmd in self.cmd:
            cmd = cmd.replace('<1>', confPath)
            os.system(cmd)