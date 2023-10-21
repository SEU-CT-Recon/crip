from crip.metric import *
from crip.io import imreadTiff

predict = imreadTiff("../Data/MetricData/rec_rabbit.tif")
gt = imreadTiff("../Data/MetricData/rec_rabbit_1.tif")

MAPE = calcMAPE(predict,gt)

PSNR = calcPSNR(predict,gt)

SSIM = calcSSIM(predict,gt)

RMSE = calcRMSE(predict,gt)

print(1+1)
