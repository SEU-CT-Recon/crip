'''
    MangoCT integration interface. See https://github.com/SEU-CT-Recon/mandoct

    https://github.com/SEU-CT-Recon/crip
'''

__all__ = ['Mgfbp', 'Mgfpj', 'MgfbpConfig', 'MgfpjConfig']

import os
import json
import tempfile
import subprocess

from .utils import cripAssert, isType
from ._typing import *


class _MgConfig(object):

    def __init__(self):
        pass

    def dumpJSON(self):
        attrs = [
            a for a in (set(dir(self)) - set(dir(object)))
            if not a.startswith('__') and not callable(getattr(self, a)) and getattr(self, a) is not None
        ]
        dict_ = dict([(k, getattr(self, k)) for k in attrs])

        return json.dumps(dict_, indent=2)

    def dumpJSONFile(self, path: str):
        with open(path, 'w') as fp:
            fp.write(self.dumpJSON())

    def fromJSON(self, json_: str):
        obj = json.loads(json_)
        for key in obj:
            self[key] = obj

    def fromJSONFile(self, path: str):
        with open(path, 'r') as fp:
            self.fromJSON(fp.read())


class MgfbpConfig(_MgConfig):

    def __init__(self):
        super().__init__()
        self.setIO(None, None, None, None, [])
        self.setGeometry(None, None, None, None, None, None, None, None)
        self.setSgmConeBeam(None, None, None, None, None, None, None)
        self.setRecConeBeam(None, None, None, None, 'HammingFilter', 1, None, None, None, None, None)

    def setIO(self,
              InputDir: str,
              OutputDir: str,
              InputFiles: str,
              OutputFilePrefix: str = '',
              OutputFileReplace: List[str] = []):
        self.InputDir = InputDir
        self.OutputDir = OutputDir
        self.InputFiles = InputFiles
        self.OutputFilePrefix = OutputFilePrefix
        cripAssert(len(OutputFileReplace) % 2 == 0, '`OutputFileReplace` should be paired.')
        self.OutputFileReplace = OutputFileReplace

    def setGeometry(self,
                    SourceIsocenterDistance: Or[int, float],
                    SourceDetectorDistance: Or[int, float],
                    TotalScanAngle: Or[int, float],
                    DetectorOffcenter: Or[int, float] = 0,
                    PMatrixFile: Or[str, None] = None,
                    SIDFile: Or[str, None] = None,
                    SDDFile: Or[str, None] = None,
                    ScanAngleFile: Or[str, None] = None,
                    DetectorOffCenterFile: Or[str, None] = None):
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
        self.setSgmFanBeam(SinogramWidth, SinogramHeight, Views, DetectorElementSize, SliceCount)
        self.ConeBeam = True
        self.SliceThickness = SliceThickness
        self.SliceOffCenter = SliceOffCenter

    def setRecFanBeam(self,
                      ImageDimension: int,
                      PixelSize: Or[int, float],
                      _Filter: str,
                      _FilterParam: Or[float, List[float]],
                      ImageRotation: Or[int, float] = 0,
                      ImageCenter: List[float] = [0, 0],
                      WaterMu: Or[float, None] = None,
                      SaveFilteredSinogram: bool = False):
        self.ImageDimension = ImageDimension
        self.PixelSize = PixelSize
        cripAssert(_Filter in ['HammingFilter', 'QuadraticFilter', 'Polynomial', 'GaussianApodizedRamp'],
                   f'Invalid _Filter: {_Filter}')
        if _Filter != 'HammingFilter':
            del self.HammingFilter
        exec(f'self.{_Filter} = _FilterParam')
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
                       ImageCenter: List[float] = [0, 0],
                       ImageCenterZ: Or[int, float] = 0,
                       WaterMu: Or[float, None] = None,
                       SaveFilteredSinogram: bool = False):
        self.setRecFanBeam(ImageDimension, PixelSize, _Filter, _FilterParam, ImageRotation, ImageCenter, WaterMu,
                           SaveFilteredSinogram)
        self.ImageSliceCount = ImageSliceCount
        self.ImageSliceThickness = ImageSliceThickness
        self.ImageCenterZ = ImageCenterZ


class MgfpjConfig(_MgConfig):

    def __init__(self):
        super().__init__()
        self.setIO(None, None, None, None, [])
        self.setGeometry(None, None, None, None)
        self.setRecConeBeam(None, None, None, None)
        self.setSgmConeBeam(None, None, None, None, None, None, None, None)

    def setIO(self,
              InputDir: str,
              OutputDir: str,
              InputFiles: str,
              OutputFilePrefix: str = '',
              OutputFileReplace: List[str] = []):
        self.InputDir = InputDir
        self.OutputDir = OutputDir
        self.InputFiles = InputFiles
        self.OutputFilePrefix = OutputFilePrefix
        cripAssert(len(OutputFileReplace) % 2 == 0, '`OutputFileReplace` should be paired.')
        self.OutputFileReplace = OutputFileReplace

    def setGeometry(self, SourceIsocenterDistance: Or[int, float], SourceDetectorDistance: Or[int, float],
                    StartAngle: Or[int, float], TotalScanAngle: Or[int, float]):
        self.SourceIsocenterDistance = SourceIsocenterDistance
        self.SourceDetectorDistance = SourceDetectorDistance
        self.StartAngle = StartAngle
        self.TotalScanAngle = TotalScanAngle

    def setRecFanBeam(self, ImageDimension: int, PixelSize: Or[int, float], SliceCount: int = 1):
        self.ConeBeam = False
        self.ImageDimension = ImageDimension
        self.PixelSize = PixelSize
        self.SliceCount = SliceCount

    def setRecConeBeam(self, ImageDimension: int, PixelSize: Or[int, float], SliceCount: int,
                       ImageSliceThickness: Or[int, float]):
        self.setRecFanBeam(ImageDimension, PixelSize, SliceCount)
        self.ConeBeam = True
        self.ImageSliceThickness = ImageSliceThickness

    def setSgmFanBeam(self,
                      Views: int,
                      DetectorElementCount: int,
                      DetectorElementSize: Or[int, float],
                      DetectorOffcenter: Or[int, float] = 0,
                      OversampleSize: int = 2):
        self.Views = Views
        self.DetectorElementCount = DetectorElementCount
        self.DetectorElementSize = DetectorElementSize
        self.DetectorOffcenter = DetectorOffcenter
        self.OversampleSize = OversampleSize

    def setSgmConeBeam(self,
                       Views: int,
                       DetectorElementCount: int,
                       DetectorElementSize: Or[int, float],
                       DetectorZElementCount: int,
                       DetectorElementHeight: Or[int, float],
                       DetectorOffcenter: Or[int, float] = 0,
                       DetectorZOffcenter: Or[int, float] = 0,
                       OversampleSize: int = 2):
        self.setSgmFanBeam(Views, DetectorElementCount, DetectorElementSize, DetectorOffcenter, OversampleSize)
        self.DetectorZElementCount = DetectorZElementCount
        self.DetectorElementHeight = DetectorElementHeight
        self.DetectorZOffcenter = DetectorZOffcenter


class _Mgbin(object):

    def __init__(self, exe: str, name: str, cudaDevice: int = 0, tempDir: str = None):
        self.exe = exe
        self.name = name
        self.cudaDevice = cudaDevice
        self.tempDir = tempDir
        self.cmd = []
        self.cmd.append([f'{self.exe}', '<1>'])

    def exec(self, conf: Or[str, _MgConfig], verbose=True):
        if isType(conf, _MgConfig):
            tmp = tempfile.NamedTemporaryFile('w',
                                              prefix='crip_mangoct_',
                                              suffix=f'.{self.name}.jsonc',
                                              dir=self.tempDir,
                                              delete=False)
            tmp.write(conf.dumpJSON())
            conf = tmp.name
            tmp.close()

        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.cudaDevice)
        stdout = stderr = None if verbose else subprocess.DEVNULL

        for cmd in self.cmd:
            if len(cmd) == 2:  # include args
                cmd[1] = cmd[1].replace('<1>', conf)
                subprocess.run(cmd, stdout=stdout, stderr=stderr)
            else:
                os.system(cmd[0])


class Mgfbp(_Mgbin):

    def __init__(self, exe: str = 'mgfbp', cudaDevice: int = 0, tempDir: str = None):
        '''
            Initialize a handler object to use the FBP tool in mangoct.
            `exe` is the path to the executable.
        '''
        super().__init__(exe, 'mgfbp', cudaDevice, tempDir)


class Mgfpj(_Mgbin):

    def __init__(self, exe: str = 'mgfpj', cudaDevice=0, tempDir: str = None) -> None:
        '''
            Initialize a handler object to use the FPJ tool in mangoct.
            `exe` is the path to the executable.
        '''
        super().__init__(exe, 'mgfpj', cudaDevice, tempDir)
