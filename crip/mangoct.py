'''
    MangoCT integration interface.

    https://github.com/SEU-CT-Recon/crip
'''

import os
import io
import sys
import json
import tempfile
import subprocess

from .utils import cripAssert, getAttrKeysOfObject, isType
from ._typing import *


class _MgCliConfig(object):
    ''' The base class for configuration of CLI version of mangoct.
    '''

    def __init__(self):
        pass

    def dumpJSON(self):
        ''' Dump the configuration to JSON string.
        '''
        dict_ = dict([(k, getattr(self, k)) for k in getAttrKeysOfObject(self)])

        return json.dumps(dict_, indent=2)

    def dumpJSONFile(self, path: str):
        ''' Dump the configuration to JSON file.
        '''
        with open(path, 'w') as fp:
            fp.write(self.dumpJSON())

    def fromJSON(self, json_: str):
        ''' Load the configuration from JSON string.
        '''
        obj = json.loads(json_)
        for key in obj:
            self[key] = obj[key]

    def fromJSONFile(self, path: str):
        ''' Load the configuration from JSON file.
        '''
        with open(path, 'r') as fp:
            self.fromJSON(fp.read())


class MgfbpCliConfig(_MgCliConfig):
    ''' Configuration class for CLI version `mgfbp`.
    '''

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


class MgfpjCliConfig(_MgCliConfig):
    ''' Configuration class for CLI version `mgfpj`.
    '''

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


class _MgCliBin(object):
    ''' The base class for execution of CLI version of mangoct.
    '''

    def __init__(self, exe: str, name: str, cudaDevice: int = 0, tempDir: str = None):
        self.exe = exe
        self.name = name
        self.cudaDevice = cudaDevice
        self.tempDir = tempDir
        self.cmd = [f'{self.exe}', '<1>']

    def exec(self, conf: Or[str, _MgCliConfig], verbose=True):
        if isType(conf, _MgCliConfig):
            tmp = tempfile.NamedTemporaryFile('w',
                                              prefix='crip_mangoct_',
                                              suffix=f'.{self.name}.jsonc',
                                              dir=self.tempDir,
                                              delete=False)
            tmp.write(conf.dumpJSON())
            conf = tmp.name
            tmp.close()

        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.cudaDevice)
        self.cmd[1] = self.cmd[1].replace('<1>', conf)
        proc = subprocess.Popen(self.cmd, stdout=subprocess.PIPE)
        for line in io.TextIOWrapper(proc.stdout, encoding="utf-8"):
            print(line, flush=True, file=sys.stdout)


class MgCliFbp(_MgCliBin):

    def __init__(self, exe='mgfbp', cudaDevice=0, tempDir: Or[str, None] = None):
        ''' Initialize a handler object to use the CLI version FBP tool in mangoct.
            `exe` is the path to the executable.
        '''
        super().__init__(exe, 'mgfbp', cudaDevice, tempDir)


class MgCliFpj(_MgCliBin):

    def __init__(self, exe='mgfpj', cudaDevice=0, tempDir: Or[str, None] = None):
        ''' Initialize a handler object to use the CLI version FPJ tool in mangoct.
            `exe` is the path to the executable.
        '''
        super().__init__(exe, 'mgfpj', cudaDevice, tempDir)


# TODO Add supports for Taichi version mangoct.
