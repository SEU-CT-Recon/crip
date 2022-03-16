'''
    MangoCT integration of crip.

    https://github.com/z0gSh1u/crip
'''

import os
import sys

from crip.utils import cripAssert


class Mgfbp:
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
            cripAssert(False, 'Unsupported platform for Mgfbp calling.')

    def exec(self, conf):
        for cmd in self.cmd:
            cmd = cmd.replace('<1>', '{}'.format(conf))
            os.system(cmd)


class Mgfpj:
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