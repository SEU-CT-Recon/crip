from crip.io import imreadTiff,imwriteTiff
from crip.shared import *
import matplotlib.pyplot as plt

primary = imreadTiff("./Data/SharedData/rec_rabbit.tif")

primary_rotate = rotate(primary,90)

primary_verticalFlip = verticalFlip(primary)
primary_horizontalFlip = horizontalFlip(primary)

primary_resize = resize(primary,[256,256])

primary_smooth = gaussianSmooth(primary,1)

primary_binning4 = binning(primary,(4,4))

plt.imshow(primary,cmap="gray")
plt.show()

plt.imshow(primary_smooth,cmap="gray")
plt.show()
