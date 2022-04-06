'''
    MangoCT integration of crip.

    [TODO] We are going to transfer `external` module to CRI. @see https://github.com/z0gSh1u/cri .

    https://github.com/z0gSh1u/crip
'''

__all__ = ['Mgfbp', 'Mgfpj']

import os
import sys
import subprocess
from crip.utils import cripAssert, sysPlatform
from crip._typing import *


class MgFBP:
    def __init__(self, exe: str, cudaDevice=0):
        '''
            Initialize a handler object to use the FBP tool in mangoct.
            `exe` is the path to the executable.
        '''
        self.exe = exe
        self.cudaDevice = cudaDevice
        self.confPath = None
        self._buildCmd()

    def _buildCmd(self):
        self.cmd = []

        platform = sysPlatform()

        if platform == 'Windows':
            self.cmd.append(f'set CUDA_VISIBLE_DEVICES={self.cudaDevice}')
        elif platform == 'Linux':
            self.cmd.append(f'export CUDA_VISIBLE_DEVICES={self.cudaDevice}')

        self.cmd.append(f'"{self.exe}" "<1>"')

    def exec(self, confPath: Or[str, None]):
        self.confPath = confPath
        cripAssert(self.confPath is not None, 'confPath for MgFBP is None.')

        for cmd in self.cmd:
            cmd = cmd.replace('<1>', '{}'.format(self.confPath))
            os.system(cmd)


class MgFPJ:
    def __init__(self, exe, cudaDevice=0) -> None:
        self.exe = exe
        self.cudaDevice = cudaDevice
        self.cmd = None
        self.buildCmd()

    def buildCmd(self):
        platform = sys.platform
        if platform.find('win32') != -1:
            self.cmd = ['set CUDA_VISIBLE_DEVICES={}'.format(self.cudaDevice), '"{}" <1>'.format(self.exe)]
        elif platform.find('linux') != -1:
            self.cmd = ['CUDA_VISIBLE_DEVICES={} "{}" <1>'.format(self.cudaDevice, self.exe)]
        else:
            cripAssert(False, 'Unsupported platform for Mgfpj calling.')

    def exec(self, conf):
        for cmd in self.cmd:
            cmd = cmd.replace('<1>', '{}'.format(conf))
            os.system(cmd)