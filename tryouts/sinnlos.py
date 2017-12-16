# import gdal
# import os
# import osr, ogr
# import subprocess
# from gdalconst import GA_ReadOnly, GA_Update
# from PlantPlanter.utils.files import deleteFile
#
#
# i = 0
# while i < 2:
#     source_extent = gdal.Open('files/streets_mask.tiff', GA_ReadOnly)
#     target_extent = gdal.Open('files/streets_clipped_temp.tiff', GA_Update)
#
#     target_extent.SetGeoTransform(source_extent.GetGeoTransform())
#     i = i + 1
#
# subprocess.call('gdalwarp files/streets_clipped_temp.tiff files/streets_clipped.tiff -t_srs "+proj=utm +zone=32 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs"')
#
# target_extent = None
# deleteFile("files/streets_clipped_temp.tiff")

from PIL import Image
import numpy as np

dop_clipped = Image.open("files/colour/streets_clipped.tiff")

np_img = np.asarray(dop_clipped)
np_img = np_img.astype(float)
np_img_noa = np_img[:, :, 0:3]
np_img_noa = np_img_noa / 255

print np_img_noa