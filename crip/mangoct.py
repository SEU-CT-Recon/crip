'''
    MangoCT integration interface. See
    https://gitee.com/njjixu/mangoct
    https://github.com/z0gSh1u/mangoct
    https://github.com/CandleHouse/mandoct

    https://github.com/z0gSh1u/crip
'''

__all__ = ['Mgfbp', 'Mgfpj', 'MgfbpConfig', 'MgfpjConfig']

import os
import json
import tempfile

from .utils import cripAssert, isType, sysPlatform
from ._typing import *


class MgfbpConfig:
    def __init__(self):
        self.setIO(None, None, None, None, None)
        self.setGeometry(None, None, None, None, None, None, None, None)
        self.setSgmConeBeam(None, None, None, None, None, None, None)
        self.setRecConeBeam(None, None, None, None, None, None, None, None, None, None, None)

    def setIO(self,
              InputDir: str,
              OutputDir: str,
              InputFile: str,
              OutputFilePrefix: str = '',
              OutputFileReplace: List[str] = []):
        self.InputDir = InputDir
        self.OutputDir = OutputDir
        self.InputFile = InputFile
        self.OutputFilePrefix = OutputFilePrefix
        cripAssert(len(OutputFileReplace) % 2 == 0, '`OutputFileReplace` should be paired.')
        self.OutputFileReplace = OutputFileReplace

    def setGeometry(self,
                    SourceIsocenterDistance: Or[int, float],
                    SourceDetectorDistance: Or[int, float],
                    TotalScanAngle: Or[int, float],
                    DetectorOffcenter: Or[int, float] = 0,
                    PMatrixFile: str = '',
                    SIDFile: str = '',
                    SDDFile: str = '',
                    ScanAngleFile: str = '',
                    DetectorOffCenterFile: str = ''):
        self.SourceIsocenterDistance = SourceIsocenterDistance
        self.SourceDetectorDistance = SourceDetectorDistance
        self.TotalScanAngle = TotalScanAngle
        self.DetectorOffcenter = DetectorOffcenter
        self.PMatrixFile = PMatrixFile
        self.SIDFile = SIDFile
        self.SDDFile = SDDFile
        self.ScanAngleFile = ScanAngleFile
        self.DetectorOffCenterFile = DetectorOffCenterFile

    def setSgmFanBeam(self,
                      SinogramWidth: int,
                      SinogramHeight: int,
                      Views: int,
                      DetectorElementSize: Or[int, float],
                      SliceCount: int = 1):
        self.ConeBeam = False
        self.SinogramWidth = SinogramWidth
        self.SinogramHeight = SinogramHeight
        self.Views = Views
        self.DetectorElementSize = DetectorElementSize
        self.SliceCount = SliceCount

    def setSgmConeBeam(self,
                       SinogramWidth: int,
                       SinogramHeight: int,
                       Views: int,
                       DetectorElementSize: Or[int, float],
                       SliceCount: int,
                       SliceThickness: Or[int, float],
                       SliceOffCenter: Or[int, float] = 0):
        self.ConeBeam = True
        self.setFanBeam(SinogramWidth, SinogramHeight, Views, DetectorElementSize, SliceCount)
        self.SliceThickness = SliceThickness
        self.SliceOffCenter = SliceOffCenter

    def setRecFanBeam(self,
                      ImageDimension: int,
                      PixelSize: Or[int, float],
                      _Filter: str,
                      _FilterParam: Or[float, List[float]],
                      ImageRotation: Or[int, float] = 0,
                      ImageCenter: List[float, float] = [0, 0],
                      WaterMu: Or[float, None] = None,
                      SaveFilteredSinogram: bool = False):
        self.ImageDimension = ImageDimension
        self.PixelSize = PixelSize
        cripAssert(_Filter in ['HammingFilter', 'QuadraticFilter', 'Polynomial', 'GaussianApodizedRamp'],
                   f'Invalid _Filter: {_Filter}')
        eval(f'self.{_Filter} = _FilterParam')
        self.ImageRotation = ImageRotation
        self.ImageCenter = ImageCenter
        self.WaterMu = WaterMu
        self.SaveFilteredSinogram = SaveFilteredSinogram

    def setRecConeBeam(self,
                       ImageDimension: int,
                       PixelSize: Or[int, float],
                       ImageSliceCount: int,
                       ImageSliceThickness: Or[int, float],
                       _Filter: str,
                       _FilterParam: Or[float, List[float]],
                       ImageRotation: Or[int, float] = 0,
                       ImageCenter: List[float, float] = [0, 0],
                       ImageCenterZ: Or[int, float] = 0,
                       WaterMu: Or[float, None] = None,
                       SaveFilteredSinogram: bool = False):
        self.setRecFanBeam(ImageDimension, PixelSize, _Filter, _FilterParam, ImageRotation, ImageCenter, WaterMu,
                           SaveFilteredSinogram)
        self.ImageSliceCount = ImageSliceCount
        self.ImageSliceThickness = ImageSliceThickness
        self.ImageCenterZ = ImageCenterZ

    def dumpJSON(self):
        attrs = [
            a for a in (set(dir(self)) - set(dir(object))) if not a.startswith('__') and not callable(getattr(self, a))
        ]
        dict_ = dict([(k, getattr(self, k)) for k in attrs])

        return json.dumps(dict_, indent=2)

    def dumpJSONFile(self, path):
        with open(path, 'w') as fp:
            fp.write(self.dumpJSON())


# TODO
class MgfpjConfig:
    def __init__(self):
        raise 'Not implemented.'


class Mgfbp:
    def __init__(self, exe: str = 'mgfbp', cudaDevice: int = 0, tempDir: str = None):
        '''
            Initialize a handler object to use the FBP tool in mangoct.
            `exe` is the path to the executable.
        '''
        self.exe = exe
        self.cudaDevice = cudaDevice
        self.tempDir = tempDir
        self.cmd = []
        self._buildCmd()

    def _buildCmd(self):
        platform = sysPlatform()

        if platform == 'Windows':
            self.cmd.append(f'set CUDA_VISIBLE_DEVICES={self.cudaDevice}')
        elif platform == 'Linux':
            self.cmd.append(f'export CUDA_VISIBLE_DEVICES={self.cudaDevice}')

        self.cmd.append(f'"{self.exe}" "<1>"')

    def exec(self, conf: Or[str, MgfbpConfig]):
        if isType(conf, MgfbpConfig):
            tmp = tempfile.NamedTemporaryFile('w', prefix='crip_mangoct', dir=self.tempDir)
            tmp.write(conf.dumpJSON())
            conf = tmp.name
            tmp.close()

        for cmd in self.cmd:
            cmd = cmd.replace('<1>', conf)
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