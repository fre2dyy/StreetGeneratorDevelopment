import os

import cv2
import numpy as np
from osgeo import gdal, ogr, osr

from files import getPathToTempFile, createFoldersForPath, saveImage, deleteFilesContainingName, deleteFile
import subprocess

from parameter_plots import ParametrizedPlot, ArrayParameter, AdjustableParameter, AdjustableArrayParameter

import scipy.ndimage
import skimage.morphology
import skimage.feature
import skimage.measure
import skimage.filters
import matplotlib.pyplot as plt
from timeit import default_timer as timer

from base_class import getSubprocessStartUpInfo


def createImageFromWMS(
		savePath,
		serverUrl,
		layers,
		boundingBox,
		pixelPerMeter=5,
		imageFormat="png",
		transparent=False,
		blockSize=(500, 500),
		removeLogos=True):
	"""
	Creates a geo tiff image at given savePath from given parameters.
	The crs of the WMS image is currently fixed at EPSG:25832.
	It also removes copyright logos if removeLogos is True.

	:param savePath:
	:type savePath str
	:param serverUrl:
	:type savePath str
	:param layers: comma separated layer names
	:type layers str
	:param boundingBox: (left, bottom, right, top) of bounding box in EPSG:25832 units.
	:type boundingBox tuple
	:param pixelPerMeter:
	:type pixelPerMeter int | float
	:param imageFormat: "jpeg", "png", etc
	:type imageFormat: str
	:param transparent:
	:type transparent: bool
	:param blockSize: (500,500) size of sub images in pixels, use multiple of pixelPerMeter
	:type blockSize: tuple
	:param removeLogos: if True, GeoProxies copyright logos will be removed
	:type removeLogos: bool
	"""
	pixelPerMeter = float(pixelPerMeter)

	createFoldersForPath(savePath)
	gdalWmsXmlPath = getPathToTempFile("gdal_wms_config.xml")

	xMin = boundingBox[0]
	xMax = boundingBox[2]
	yMin = boundingBox[1]
	yMax = boundingBox[3]
	widthInMeter = xMax - xMin
	heightInMeter = yMax - yMin
	imageWidth = int(round(widthInMeter * pixelPerMeter))
	imageHeight = int(round(heightInMeter * pixelPerMeter))

	# if blockSize is None:
	# 	# max block size given by server is 5000x5000, we use min block Size 500x500
	# 	blockWidth = int(round(min(max(imageWidth / 4, 500), 5000)))
	# 	blockHeight = int(round(min(max(imageWidth / 4, 500), 5000)))
	# 	blockSize = (blockHeight, blockWidth)

	name, extension = os.path.splitext(savePath)
	if extension.lower() == ".jpeg" or extension.lower() == ".jpg":
		gdalImageFormat = "JPEG"
	elif extension.lower() == ".png":
		gdalImageFormat = "PNG"
	else:
		gdalImageFormat = "GTiff"

	createGdalWmsXml(
		gdalWmsXmlPath,
		serverUrl=serverUrl,
		layers=layers,
		bb=boundingBox,
		size=(imageWidth, imageHeight),
		blockSize=blockSize,
		imageFormat=("image/" + imageFormat),
		transparent=transparent
	)
	subprocess.call(
		[
			"gdal_translate",
			"-of", gdalImageFormat,
			"-outsize", str(imageWidth), str(imageHeight),
			gdalWmsXmlPath, savePath
		],
		# startupinfo=getSubprocessStartUpInfo()
	)

	if removeLogos is True and "geoproxy.geoportal-th.de" in serverUrl:
		# At each position of a logo, a new image will be loaded from the WMS (logoWidth, 2 * logoHeight).
		imageDataSet = gdal.Open(savePath, gdal.GA_Update)  # type: gdal.Dataset
		imageData = np.array(imageDataSet.ReadAsArray(), dtype='uint8')  # [R,G,B,(A)][y][x]
		blockWidth = blockSize[0]
		blockHeight = blockSize[1]
		blockCountX = int(np.ceil(float(imageWidth) / blockWidth))
		blockCountY = int(np.ceil(float(imageHeight) / blockHeight))
		blockWidthInMeter = blockWidth / pixelPerMeter
		blockHeightInMeter = blockHeight / pixelPerMeter
		logoWidth = 65
		logoHeight = 25
		logoWidthInMeter = logoWidth / pixelPerMeter
		logoHeightInMeter = logoHeight / pixelPerMeter

		for y in range(0, blockCountY):
			for x in range(0, blockCountX):
				if y < blockCountY - 1:
					bb = (
						xMin + x * blockWidthInMeter,  # left
						yMax - blockHeightInMeter + logoHeightInMeter - y * blockHeightInMeter,  # top
						xMin + logoWidthInMeter + x * blockWidthInMeter,  # right
						yMax - blockHeightInMeter - logoHeightInMeter - y * blockHeightInMeter  # bottom
					)
				else:  # the last blocks in y direction maybe not complete blocks, so use yMin
					bb = (
						xMin + x * blockWidthInMeter,  # left
						yMin + logoHeightInMeter,  # top
						xMin + logoWidthInMeter + x * blockWidthInMeter,  # right
						yMin - logoHeightInMeter  # bottom
					)
				blockTempPath = getPathToTempFile("temp_block.tiff")
				createGdalWmsXml(
					gdalWmsXmlPath,
					serverUrl=serverUrl,
					layers=layers,
					bb=bb,
					size=(logoWidth, logoHeight * 2),
					blockSize=(logoWidth, logoHeight * 2),
					imageFormat=("image/" + imageFormat),
					transparent=transparent
				)
				# suppress showing up console window
				subprocess.call(
					[
						"gdal_translate",
						"-of", gdalImageFormat,
						"-outsize", str(logoWidth), str(logoHeight * 2),
						gdalWmsXmlPath, blockTempPath
					],
					startupinfo=getSubprocessStartUpInfo()
				)

				# remove the logo in overlay image
				overlayDataSet = gdal.Open(blockTempPath, gdal.GA_ReadOnly)  # type: gdal.Dataset
				overlayData = np.array(overlayDataSet.ReadAsArray(), dtype='uint8')  # [R,G,B][y][x]
				overlayData = removeLogo(overlayData)

				# set the overlay data at position where logos are in original image
				if y < blockCountY - 1:
					logoData = imageData[:,
							   blockHeight - logoHeight + y * blockHeight:blockHeight + y * blockHeight,
							   x * blockWidth:x * blockWidth + logoWidth
							   ]
				else:  # the last blocks in y direction maybe not complete blocks, so use imageHeight
					logoData = imageData[:,
							   imageHeight - logoHeight:imageHeight,
							   x * blockWidth:x * blockWidth + logoWidth
							   ]
				# check if image is big enough
				clippedOverlayData = overlayData[:, 0:logoData.shape[1], 0:logoData.shape[2]]

				if y < blockCountY - 1:
					imageData[:,
					blockHeight - logoHeight + y * blockHeight:blockHeight + y * blockHeight,
					x * blockWidth:x * blockWidth + logoWidth
					] = clippedOverlayData
				else:  # the last blocks in y direction maybe not complete blocks, so use imageHeight
					imageData[:,
					imageHeight - logoHeight:imageHeight,
					x * blockWidth:x * blockWidth + logoWidth
					] = clippedOverlayData

				# cleanup
				overlayDataSet = None
				deleteFilesContainingName(os.path.basename(blockTempPath))

		for i in range(1, imageDataSet.RasterCount + 1):
			imageDataSet.GetRasterBand(i).WriteArray(imageData[i - 1])  # causes error when closing app window
		imageDataSet = None


def createGdalWmsXml(path,
					 serverUrl,
					 layers,
					 size,
					 version="1.3.0",
					 srs="EPSG:25832",
					 crs="EPSG:25832",
					 imageFormat="image/jpeg",
					 bb=(635033, 5615730, 635211, 5615791),
					 projection="EPSG:25832",
					 blockSize=(512, 512),
					 bandsCount=3,
					 transparent=False
					 ):
	"""
	Creates a xml file for gdal to connect to a wms.

	http://www.gdal.org/frmt_wms.html

	:param path: path where the xml will be created
	:type path: str
	:param serverUrl: url to wms service
	:type serverUrl: str
	:param layers: comma separated layer names
	:type layers: str
	:param size: (X, Y) pixel size of image
	:type size: tuple
	:param version: version of the wms service
	:type version: str
	:param srs:
	:type srs: str
	:param crs:
	:type crs: str
	:param imageFormat:
	:type imageFormat: str
	:param bb: (left, bottom, right, top) of bounding box in srs units.
	:type bb: tuple
	:param projection:
	:type projection: str
	:param blockSize: (X, Y) pixel size of blocks (where to split image so multiple calls can be made)
	:type blockSize: tuple
	:param bandsCount:
	:type bandsCount: int | str
	:param transparent
	:type transparent: bool



	:return:
	:rtype: str
	"""
	if transparent is True:
		bandsCount += 1
	gdalWmsXmlTemplate = """<GDAL_WMS>
	<Service name="WMS">
		<Version>""" + version + """</Version>
		<ServerUrl>""" + serverUrl + """</ServerUrl>
		<SRS>""" + srs + """</SRS>
		<CRS>""" + crs + """</CRS>
		<ImageFormat>""" + imageFormat + """</ImageFormat>
		<Layers>""" + layers + """</Layers>
		<Transparent>""" + ("TRUE" if transparent is True else "FALSE") + """</Transparent>
	</Service>
	<DataWindow>
		<UpperLeftX>""" + str(bb[0]) + """</UpperLeftX>
		<UpperLeftY>""" + str(bb[3]) + """</UpperLeftY>
		<LowerRightX>""" + str(bb[2]) + """</LowerRightX>
		<LowerRightY>""" + str(bb[1]) + """</LowerRightY>
		<SizeX>""" + str(size[0]) + """</SizeX>
		<SizeY>""" + str(size[1]) + """</SizeY>
	</DataWindow>
	<Projection>""" + projection + """</Projection>
	<BlockSizeX>""" + str(blockSize[0]) + """</BlockSizeX>
	<BlockSizeY>""" + str(blockSize[1]) + """</BlockSizeY>
	<BandsCount>""" + str(bandsCount) + """</BandsCount>
</GDAL_WMS>"""
	# <TileLevel>""" + str(tileLevel) + """</TileLevel>
	# <TileCountX>""" + str(tileCount[0]) + """</TileCountX>
	# <TileCountY>""" + str(tileCount[1]) + """</TileCountY>
	out = open(path, 'wb')
	out.write(gdalWmsXmlTemplate)
	out.close()


def opening(data, kernel=np.ones((3, 3)), iterations=1, plot=False):
	"""
	Morphological opening using opencv.

	:param data: image
	:type data: np.ndarary
	:param kernel: structure element
	:type kernel: data: np.ndarary
	:param iterations:
	:type iterations: int
	:param plot:
	:type plot: bool
	:return: opened image
	:rtype: np.ndarray
	"""
	ero = cv2.erode(data, kernel, iterations=iterations)
	dil = cv2.dilate(ero, kernel, iterations=iterations)
	if plot:
		ParametrizedPlot(
			opening,
			parameters=[ArrayParameter("data", data, plot=True)],
			adjustableParameters=[
				AdjustableArrayParameter("kernel", int(kernel.shape[0]), 1, 100),
				AdjustableParameter("iterations", iterations, 0, 10)]
		)
	return dil


def closing(data, kernel=np.ones((3, 3)), iterations=1, plot=False):
	"""
	Morphological closing using opencv.

	:param data: image
	:type data: np.ndarary
	:param kernel: structure element
	:type kernel: data: np.ndarary
	:param iterations:
	:type iterations: int
	:param plot:
	:type plot: bool
	:return: closed image
	:rtype: np.ndarray
	"""
	dil = cv2.dilate(data, kernel, iterations=iterations)
	ero = cv2.erode(dil, kernel, iterations=iterations)
	if plot:
		ParametrizedPlot(
			closing,
			parameters=[ArrayParameter("data", data, plot=True)],
			adjustableParameters=[
				AdjustableArrayParameter("kernel", int(kernel.shape[0]), 1, 100),
				AdjustableParameter("iterations", iterations, 0, 10)]
		)
	return ero


def hitOrMiss(src, hit, miss=None, dst=None):
	"""
	Hit or Miss transform with open cv.
	Based on http://opencv-code.com/tutorials/hit-or-miss-transform-in-opencv/

	:param src: image on which transform will be done
	:type src: np.ndarray
	:param hit: hit mask
	:type hit: np.ndarray
	:param miss: miss mask, if none, inverted hit mask will be used
	:type miss: np.ndarray
	:param dst:
	:type dst: np.ndarray

	:return:
	:rtype: np.ndarray
	"""
	if src.dtype != np.uint8:
		src = src.astype(np.uint8)
	if hit.dtype != np.uint8:
		hit = hit.astype(np.uint8)
	if miss is None:
		miss = 1 - hit
	if miss != np.uint8:
		miss = miss.astype(np.uint8)

	cv2.normalize(src, dst=src, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

	e1 = cv2.erode(src, hit, iterations=1, borderType=cv2.BORDER_CONSTANT, borderValue=0)
	e2 = cv2.erode(1 - src, miss, iterations=1, borderType=cv2.BORDER_CONSTANT, borderValue=0)
	if isinstance(dst, np.ndarray):
		dst = np.logical_and(e1, e2)
	else:
		return np.logical_and(e1, e2)


def getLineEndings(img):
	"""
	Find endings of lines in an image (pixel size of lines must be 1)
	Based on ENDPOINTS DETERMINATION in http://www.ias-iss.org/ojs/IAS/article/viewFile/862/765

	:param img:
	:type img: np.ndarray

	:return
	:rtype: np.ndarray
	"""
	neighbour4 = np.array([[0, 1, 0],
						   [0, 1, 0],
						   [0, 0, 0]])
	neighbour8 = np.array([[0, 0, 1],
						   [0, 1, 0],
						   [0, 0, 0]])
	ret = np.zeros(img.shape, dtype=np.uint8)
	for i in range(0, 4):
		hitMiss4 = hitOrMiss(img, neighbour4)
		ret = np.logical_or(ret, hitMiss4)
		neighbour4 = np.rot90(neighbour4)

		hitMiss8 = hitOrMiss(img, neighbour8)
		ret = np.logical_or(ret, hitMiss8)
		neighbour8 = np.rot90(neighbour8)
	return ret


def removeLogo(img):
	"""
	Removes 25 pixel in height where copyright is.

	:param img: [Channels][y][x] or [y][x]
	:type img: np.ndarray

	:rtype: np.ndarray
	"""
	if img.ndim == 2:
		return img[0:img.shape[0] - 25, :]
	elif img.ndim == 3:
		return img[:, 0:img.shape[1] - 25, :]


def scaleToUInt(data, dtype=np.uint8, copy=True):
	"""
	Scales given numpy array values to 0-maxVal.
	Using formula from "Computer Processing of Remotely-Sensed Images - An Introduction" P.98.

	:param data:
	:type data: np.ndarray
	:type dtype: type
	:param copy: if true, a copy will be returned, else the input data will be changed
	:type copy: bool

	:rtype: np.ndarray
	"""
	minValue = data.min()
	maxValue = data.max()
	if not (np.iinfo(dtype).max == maxValue and minValue == 0):
		data = data.astype(np.float64, copy=copy)
		data = (((data - minValue) / (maxValue - minValue)) * np.iinfo(dtype).max).astype(dtype)
	else:
		if copy is True:
			data = data.astype(dtype)
	return data


def pixel2coord(geoTransform, x, y):
	"""
	Returns the coordinates of a pixel position x,y in the image space of geoTransform.

	:param geoTransform: a gdal geoTransform (ulx, x pixel resolution, 0, uly, 0, -y pixel resolution)
						 @see http://www.gdal.org/gdal_tutorial.html#gdal_tutorial_dataset
	:type geoTransform: tuple
	:param x:
	:type x: int | float
	:param y:
	:type y: int | float
	"""
	originX = geoTransform[0]
	originY = geoTransform[3]
	pixelWidth = geoTransform[1]
	pixelHeight = geoTransform[5]
	coordX = originX + (pixelWidth * x)
	coordY = originY + (pixelHeight * y)
	return coordX, coordY


def coord2pixel(geoTransform, coordX, coordY):
	"""
	Returns the pixel position in the image space of a coordinates.

	:param geoTransform: a gdal geoTransform (ulx, x pixel resolution, 0, uly, 0, -y pixel resolution)
						 @see http://www.gdal.org/gdal_tutorial.html#gdal_tutorial_dataset
	:type geoTransform: tuple
	:param coordX:
	:type coordX: float
	:param coordY:
	:type coordY: float
	"""
	originX = geoTransform[0]
	originY = geoTransform[3]
	pixelWidth = geoTransform[1]
	pixelHeight = geoTransform[5]
	x = int((coordX - originX) / pixelWidth)
	y = int((coordY - originY) / pixelHeight)
	return x, y


def convertRasterToVectorPolygons(dataPath, savePath, returnDataSource=False):
	"""
	Creates a vector shp file from given image file.

	It must be a saved image because gdals polygonize will only operate with GDALRasterBandShadow.

	:param dataPath: path where image to convert is.
	:type dataPath: str
	:param savePath: where shape file will be saved (ending .shp)
	:type savePath: str
	:param returnDataSource: if True, the shape file data source will be returned, else it will be closed
	:type returnDataSource: bool

	:return:
	:rtype: ogr.DataSource
	"""
	dataSet = gdal.Open(dataPath, gdal.GA_ReadOnly)
	vectorDriver = ogr.GetDriverByName("ESRI Shapefile")  # type: ogr.Driver
	deleteFile(savePath)
	dataSource = vectorDriver.CreateDataSource(savePath)  # type: ogr.DataSource
	srs = osr.SpatialReference()
	srs.ImportFromWkt(dataSet.GetProjection())
	layer = dataSource.CreateLayer("polygonize", srs=srs)  # type: ogr.Layer

	# create feature with attribute "DN", where polygonize will write in the original color a polygon was created from
	fieldName = 'DN'
	fd = ogr.FieldDefn(fieldName, ogr.OFTInteger)
	layer.CreateField(fd)
	fieldId = 0

	gdal.Polygonize(dataSet.GetRasterBand(1), None, layer, fieldId, callback=None)

	dataSet = None
	if returnDataSource is True:
		return dataSource
	else:
		dataSource = None


def entropy(data, window=np.ones((3, 3)), mask=None, plot=False):
	"""
	Calculates entropy of given data in a window for each pixel in image.

	:type data: np.ndarray
	:param window: neighborhood of each pixel which included for entropy calculation.
	:type window: np.ndarray
	:param mask: must be size of image, all values greater 0 will be used for entropy calculation
	:type mask: None | np.ndarray
	:type plot: bool

	:return:
	:rtype: np.ndarray
	"""

	start = timer()
	entro = skimage.filters.rank.entropy(data, selem=window, mask=mask)
	print "entropy: %4.3f" % (timer() - start)
	if plot:
		ParametrizedPlot(
			entropy,
			parameters=[
				ArrayParameter("data", data, plot=True),
				ArrayParameter("mask", mask)
			],
			adjustableParameters=[AdjustableArrayParameter("window", int(window.shape[0]), 1, 100)]
		)
		plt.show(block=False)
	return entro


def reclassify(data, classCount, plot=False):
	"""
	Given data will be split up into equal intervals. The number of intervals is given by classCount.

	:param data: image data [y][x]
	:type data: np.ndarray
	:param classCount:
	:type classCount: int
	:type plot: bool

	:return:
	:rtype: np.ndarray
	"""
	hist, binEdges = np.histogram(data, classCount)
	reclassifiedData = np.zeros(data.shape, dtype=np.uint8)
	for i, edge in enumerate(binEdges):
		if (i + 1) < len(binEdges):
			reclassifiedData[(binEdges[i] <= data) & (data <= binEdges[i + 1])] = i
	if plot:
		ParametrizedPlot(
			reclassify,
			parameters=[ArrayParameter("data", data, copy=True, plot=True)],
			adjustableParameters=[AdjustableParameter("classCount", classCount, 0, max(100, classCount))],
			showHist=True
		)
		plt.show(block=False)
	return reclassifiedData


def combineMasks(masks):
	"""
	Combines cv2.bitwise_or the given (binary [0,255]) masks.
	Masks must have same size!

	:param masks: list of mask (can be np.ndarray or str)
	:type masks: list[np.ndarray | str]

	:return: combined binary mask of given mask
	:rtype: np.ndarray
	"""

	# read files if masks are paths to files
	unableToOpenIds = []
	for i in range(0, len(masks)):
		if not isinstance(masks[i], np.ndarray):
			maskDataSet = gdal.Open(masks[i], gdal.GA_ReadOnly)  # type: gdal.Dataset
			if maskDataSet is not None:
				masks[i] = np.array(maskDataSet.ReadAsArray(), dtype=np.uint8)  # [y][x]
				maskDataSet = None
			else:
				unableToOpenIds.append(i)
	unableToOpenIds.sort(reverse=True)
	for i in unableToOpenIds:
		del masks[i]

	combinedMask = np.zeros_like(masks[0])
	for mask in masks:  # type: np.ndarray
		if mask.shape != combinedMask.shape:
			raise ValueError("Masks must have same size!")
		combinedMask = cv2.bitwise_or(combinedMask, mask)

	return combinedMask


def scaleImageToSize(inputPath, outputPath, resolutionX, resolutionY, interpolation=gdal.GRA_NearestNeighbour):
	"""
	Scales input image to sizeX x sizeY.

	:param inputPath:
	:type inputPath: str
	:param outputPath:
	:type outputPath: str
	:param resolutionX: pixel resolution in X
	:type resolutionX: int
	:param resolutionY: pixel resolution in Y
	:type resolutionY: int
	:param interpolation: ie. gdal.GRA_NearestNeighbour or "near"
	:type interpolation: int | str
	"""
	if not isinstance(interpolation, str):
		interpolation = getGDALInterpolationName(interpolation)
	subprocess.call(
		[
			"gdalwarp",
			"-of", "GTiff",
			"-ts", str(resolutionX), str(resolutionY),
			"-r", interpolation,
			inputPath,
			outputPath
		],
		startupinfo=getSubprocessStartUpInfo()
	)


def getGDALInterpolationName(interpolation):
	"""
	Returns the name of a gdal interpolation type.

	@see http://www.gdal.org/gdalwarp.html

	:param interpolation: i.e. gdal.GRA_NearestNeighbour
	:type interpolation: int

	:rtype: str
	"""
	if interpolation == gdal.GRA_NearestNeighbour:
		return "near"
	if interpolation == gdal.GRA_Bilinear:
		return "bilinear"
	if interpolation == gdal.GRA_Cubic:
		return "cubic"
	if interpolation == gdal.GRA_CubicSpline:
		return "cubicspline"
	if interpolation == gdal.GRA_Lanczos:
		return "lanczos"
	if interpolation == gdal.GRA_Average:
		return "average"
	if interpolation == gdal.GRA_Mode:
		return "mode"

	return "near"


def combineToMultiChannelImage(images):
	"""
	Combines given images (paths or np.ndarrays) to  M x N x len(images) array
	Images must have same size!

	:param images: list of 8bit images (can be np.ndarray or str)
	:type images: list

	:return:
	:rtype: np.ndarray
	"""

	# read files if masks are paths to files
	unableToOpenIds = []
	for i in range(0, len(images)):
		if not isinstance(images[i], np.ndarray):
			imageDataSet = gdal.Open(images[i], gdal.GA_ReadOnly)  # type: gdal.Dataset
			if imageDataSet is not None:
				images[i] = np.array(imageDataSet.ReadAsArray(), dtype=np.uint8)  # [y][x]
				imageDataSet = None
			else:
				unableToOpenIds.append(i)

	unableToOpenIds.sort(reverse=True)
	for i in unableToOpenIds:
		del images[i]

	# check that all images have save size
	imageSize = images[0].shape
	for image in images:  # type: np.ndarray
		if image.shape != imageSize:
			raise ValueError("Images must have same size!")

	return np.array(images, np.uint8)


def setInfiniteValuesToFinite(data):
	"""
	Sets NaNs to 0, inf to the max finite value and sets -inf to the min finite value.

	:type data: np.ndarray

	:rtype: np.ndarray
	"""
	data[np.isnan(data)] = 0  # 0/0 is nan
	finiteValues = np.isfinite(data)  # non inf values
	if not finiteValues.all():
		data[np.isposinf(data)] = data[finiteValues].max()  # 1/0 is inf
		data[np.isneginf(data)] = data[finiteValues].min()  # -1/0 is -inf
	return data


def getLocalMaxima(image, minDistance, seType=None):
	"""
	Returns local maxima in image.
	Based on http://stackoverflow.com/questions/5550290/find-local-maxima-in-grayscale-image-using-opencv/21023493#21023493

	The image will be dilated with structure element.

	:param image:
	:type image: np.ndarray
	:param minDistance: size of the structure element
	:type minDistance: int
	:param seType: structure element type i.e. cv2.MORPH_RECT
	:type seType: int

	:return: image with local maximas
	:rtype: np.ndarray
	"""
	size = 2 * minDistance + 1
	if seType is None:
		seType = cv2.MORPH_RECT
	kernel = cv2.getStructuringElement(seType, (size, size))
	dil = cv2.dilate(image, kernel=kernel)
	# no changes in
	mask = cv2.compare(image, dil, cmpop=cv2.CMP_EQ)
	localMaxiImage = cv2.compare(image, 0, cmpop=cv2.CMP_GT)
	localMaxiImage[mask == 0] = 0

	# Remove the image borders
	localMaxiImage[:minDistance] = 0
	localMaxiImage[-minDistance:] = 0
	localMaxiImage[:, :minDistance] = 0
	localMaxiImage[:, -minDistance:] = 0
	return localMaxiImage
