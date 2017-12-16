import sip
for api in ["QDate", "QDateTime", "QString", "QTextStream", "QTime", "QUrl", "QVariant"]:
    sip.setapi(api, 2)

import numpy as np
from osgeo import gdal
import scipy.ndimage
import skimage.transform
import skimage.measure
import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageEnhance

from utils import plant_detection, files, raster, parameter_plots
from utils.parameter_plots import ParametrizedPlot, ArrayParameter, AdjustableParameter, AdjustableArrayParameter, \
	Parameter
from utils.plant_detection import Pixelclassification
from utils.vector import clipping, CopyExtentandEPSG
from utils.miscellaneous import imagetonumpy, numpytoimage, change_contrast


def main():
	#parameter_plots.init()
	colorImagePath = "files/colour/streets_clipped.tiff"
	cirImagePath = "files/cir/streets_clipped.tiff"
	colorImageDataSet = gdal.Open(colorImagePath, gdal.GA_ReadOnly)  # type: gdal.Dataset
	colorData = np.array(colorImageDataSet.ReadAsArray(), dtype='uint8')  # [R,G,B][y][x]
	r = colorData[0]  # type: np.ndarray
	g = colorData[1]  # type: np.ndarray
	b = colorData[2]  # type: np.ndarray
	cirImageDataSet = gdal.Open(cirImagePath, gdal.GA_ReadOnly)  # type: gdal.Dataset
	cirData = np.array(cirImageDataSet.ReadAsArray(), dtype='uint8')  # [NIR,R,G][y][x]
	nir = cirData[0]  # type: np.ndarray

	# shadows by ORIGINAL formula from Ono (all must be divided by number of channels N
	r_norm = nir.astype(np.float32) / ((r.astype(np.float32) + g.astype(np.float32) + b.astype(np.float32) + nir.astype(np.float32) )/4.0)
	si_new = r_norm - nir
	si_new = r.astype(np.float32) / ((r.astype(np.float32) + g.astype(np.float32) + b.astype(np.float32)  )/3.0)
	#si_new = r_norm - r



	si_value = 0.85		# ?best value?: 0.85

	si_new2 = si_new[:]
	si_new2 = np.repeat(si_new2[:, :, np.newaxis], 3, axis=2)	# copy values of 1st dimension to 2nd and 3rd dimension


	# create shadow mask
	i = 0
	j = 0
	while i < si_new2.shape[0]:
		while j < si_new2.shape[1]:
			if si_new2[i,j,0] <= si_value and si_new2[i,j,0] > 0:
				si_new2[i,j,:] = 1
			else:
				si_new2[i,j,:] = 0

			j = j + 1
		j = 0
		i = i + 1

	si_new2 = si_new2.astype(int)
	si_new2 = si_new2 * 255
	si_new2_mask = Image.fromarray(si_new2.astype(np.uint8))

	si_new2_mask.save("files/shadow/streets_shadow_mask_temp.tiff")


	# eliminate white borders on the shadow edges
	si_new2_mask = cv2.imread("files/shadow/streets_shadow_mask_temp.tiff")
	imgray_a = cv2.cvtColor(si_new2_mask, cv2.COLOR_BGR2GRAY)
	ret_a, thresh_a = cv2.threshold(imgray_a, 127, 255, 0)

	_, ca, _ = cv2.findContours(thresh_a, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
	print "shape: ", np.shape(ca[0])
	cv2.drawContours(si_new2_mask, ca, -1, (0, 0, 0), 1)

	streets_shadow_mask_final = "files/shadow/streets_shadow_mask_final.tiff"
	cv2.imwrite(streets_shadow_mask_final, si_new2_mask)


	dop = ("files/colour/streets_dop.tif")
	shadowMask = (streets_shadow_mask_final)
	clippedPathTemp = ("files/shadow/streets_shadow_clipped_temp.tiff")

	clipping(shadowMask, dop, clippedPathTemp)

	# correct brightness and colour of selected shadows
	shadow_temp = Image.open("files/shadow/streets_shadow_clipped_temp.tiff")
	shadow_temp = ImageEnhance.Brightness(shadow_temp).enhance(3.0)
	shadow_temp = ImageEnhance.Color(shadow_temp).enhance(0.1)
	shadow_temp.save("files/shadow/streets_shadow_imageenhance.tiff")


	shadow_np = imagetonumpy(shadow_temp)
	dop_np = Image.open(colorImagePath)
	dop_np = imagetonumpy(dop_np)


	# put corrected shadow parts together with "no shadow"-parts of the image
	i = 0
	j = 0
	while i < shadow_np.shape[0]:
		while j < shadow_np.shape[1]:

			if shadow_np[i, j, 0] == 0 and shadow_np[i, j, 1] == 0 and shadow_np[i, j, 2] == 0:
				dop_np[i, j, :] = dop_np[i, j, :]
			else:
				dop_np[i, j, :] = shadow_np[i, j, :]

			j = j + 1
		j = 0
		i = i + 1


	final = numpytoimage(dop_np)
	#final = change_contrast(final, 130)		# raise contrast
	final.save("files/shadow/streets_shadow.tiff")


	# Copy extent and EPSG from input raster to output raster
	EEPSG = ("files/colour/streets_mask.tiff")
	noEEPSG = ("files/shadow/streets_shadow.tiff")
	targetPath = ("files/shadow/streets_shadow_EPSG.tiff")

	CopyExtentandEPSG(EEPSG, noEEPSG, targetPath)

	# colorImageDataSet = None
	# colorData = None

if __name__ == "__main__":
	main()
