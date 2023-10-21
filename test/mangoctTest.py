import os
from os.path import abspath
from crip.mangoct import MgfbpConfig, Mgfbp

# 建立mgfbp运行需要的文件夹
# os.makedirs('./sgm', exist_ok=1)
# os.makedirs('./rec', exist_ok=1)
# os.makedirs('./tmp', exist_ok=1)

#重建参数
SID = 311.14
SDD = 435.94
TotalAngle = 360
OffsetU = 0.14

SinogramWidth = 768
SinogramHeight = 486
TotalViews = 400
DetectorElementSize = SliceThickness = 0.2992
OffsetV = 1.03

ImageDimension = 512
PixelSize = ImageSliceThickness = 0.3125
ImageSliceCount = 256
Filter = 'GaussianApodizedRamp'
FilterParam = 1

print("-----start recon-----")
cfg = MgfbpConfig()
cfg.setIO(abspath('./sgm'), abspath('./rec'), 'sgm.*', OutputFileReplace=['sgm', 'rec'])
cfg.setGeometry(SID, SDD, TotalAngle, OffsetU)
cfg.setSgmConeBeam(SinogramWidth, TotalViews, TotalViews, DetectorElementSize, SinogramHeight, SliceThickness, OffsetV)
cfg.setRecConeBeam(ImageDimension, PixelSize, ImageSliceCount, ImageSliceThickness, Filter, FilterParam, ImageRotation=90)
Mgfbp(tempDir='./tmp').exec(cfg)
print("-----finish recon-----")