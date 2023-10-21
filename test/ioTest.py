from crip.io import *
import numpy as np

# listDirectory(folder,style="filename")获取文件名或路径
folder = r"Data\IoData"
# style常用的两种：filename和fullpath。默认的是filename
content1 = listDirectory(folder, style="filename")
# ['LI_XIANG.CT.ABDOMEN_11_ABDOMEN_DE_BT_(ADULT).0010.0001.2017.08.30.09.57.48.421875.782907008.IMA',
# 'al_gt.tif', 'b_0.raw', 'write_al_gt.tif', 'write_b_0.raw']
content2 = listDirectory(folder, style="fullpath")
# ['Data\\IoData\\LI_XIANG.CT.ABDOMEN_11_ABDOMEN_DE_BT_(ADULT).0010.0001.2017.08.30.09.57.48.421875.782907008.IMA',
# 'Data\\IoData\\al_gt.tif','Data\\IoData\\b_0.raw','Data\\IoData\\write_al_gt.tif','Data\\IoData\\write_b_0.raw']

# imreadRaw(raw_path,H,W,nSlice=1)读取raw文件
raw_path = r'../Data/IoData/raws/b_0.raw'
raw = imreadRaw(raw_path, 512, 512, dtype=np.float32 ,nSlice=1)  # raw里面存放着512*512的数据，类型是ndarray raw:(512,512)
raw1 = imreadRaw(raw_path, 256, 256,dtype=np.float32, nSlice=4)  # 如果将512*512的一张图片的数据看成256*256数据的话，那么在这个raw文件其实就是存放了4张256*256的数据 raw1:(4,12,512), nslice作为深度通道C

# imreadRaws(raw_path,H,W,nSlice=1)读取文件夹里面所有的raw文件
raw_dir_path = "../Data/IoData/raws"
raws = imreadRaws(raw_dir_path, 512, 512, dtype=np.float32, nSlice=1)

# imwriteRaw(raw,save_raw_path)写入raw文件
raw_path = r'../Data/IoData/raws/b_0.raw'
raw = imreadRaw(raw_path, 512, 512,dtype=np.float32, nSlice=1)  # raw里面存放着512*512的数据，类型是ndarray
save_raw_path = "../Data/IoData/raws/write_b_0.raw"
imwriteRaw(raw, save_raw_path)

# 我们更加愿意使用tif文件，因为在打开ImageJ的时候不需要去设置width和height
# imreadTiff(tiff_path)读入tiff文件
tiff_path = r"../Data/IoData/tifs/al_gt.tif"
tiff = imreadTiff(tiff_path)

# imreadTiffs()读取文件夹里面所有的tif文件
tif_dir_path = "../Data/IoData/tifs"
tifs = imreadTiffs(tif_dir_path,dtype=np.float32)

# imwriteTiff(tiff,save_tiff_path)写入tiff文件
tiff_path = r"../Data/IoData/tifs/al_gt.tif"
tiff = imreadTiff(tiff_path)
save_tiff_path = "../Data/IoData/tifs/write_al_gt.tif"
imwriteTiff(tiff, save_tiff_path)

# imreadDicom(DICOM_path)读入DICOM文件:将图像数据、相关扫描以及重建参数按照DICOM标准封装成最终图像。
DICOM_path = r"../Data/IoData/dicoms/1-01.dcm"
DICOM = imreadDicom(DICOM_path)

# imreadTiffs()读取文件夹里面所有的Dicom文件
DICOM_dir_path = "../Data/IoData/dicoms"
dicoms = imreadDicoms(DICOM_dir_path)


param = fetchCTParam(dicoms,key='Slice Thickness')
print(1+1)