import os
import numpy as np
from crip.io import imreadDicoms, imwriteTiff, listDirectory
from crip.postprocess import fovCrop
from crip.de import genMaterialPhantom

# 全数据集路径
dir_ = '/mnt/new_no1/zhuoxu/AAPM-LowDoseCT/Reconstruction'
# 输出路径
outDir = '/mnt/new_no1/zhuoxu/DEREDCNN/AAPM-Workspace/AAPM-L-Legacy/AAPM-Basis'
lowest = 230  # 最低置信阈值
range_ = 100  # 置信范围
gaussianSimga = 1  # 高斯平滑
zsmooth = 5  # 切片平均
boneBase = 1200 + 1000  # 骨基准HU

for path, case in listDirectory(dir_, style='both'):
    if case.startswith('L'):
        print('正在处理', case)
        path = os.path.join(path, listDirectory(path)[0])

        _idx = 0
        for _idx in [0, 1]:
            if 'Full dose' in os.path.join(path, listDirectory(path)[0]):
                break
        path = os.path.join(path, listDirectory(path)[_idx])  # Full Dose

        imgs = imreadDicoms(path, np.float32)
        assert np.min(imgs) >= 0  # HU + 1000
        imgs = fovCrop(imgs, 256) - 1000  # to HU
        # imgs = imgs[len(imgs) // 2:]  # rule out lung

        water, bone = genMaterialPhantom(imgs, zsmooth, gaussianSimga, lowest, lowest + range_, boneBase=boneBase)
        out = os.path.join(outDir, case)
        os.makedirs(out, exist_ok=True)

        imwriteTiff(water, os.path.join(out, 'water.tif'))
        imwriteTiff(bone, os.path.join(out, 'bone.tif'))
