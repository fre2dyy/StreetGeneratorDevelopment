import math
import os
from timeit import default_timer as timer

import cv2
import matplotlib.pyplot as plt
import numpy as np
from osgeo import gdal, ogr, osr
from sklearn.cluster import KMeans

from base_class import BaseClass
from files import deleteFile
from parameter_plots import ParametrizedPlot, AdjustableParameter, ArrayParameter
from qgis_utils import createPointsAlongLines
from raster import closing, opening, coord2pixel, pixel2coord, entropy, scaleToUInt, reclassify, setInfiniteValuesToFinite, \
	getLocalMaxima
from vector import createBuffers, convertPolygonsToLines, getPointsGreaterThanThresholdInRasterImage


class Pixelclassification(BaseClass):
	"""
	Class with methods to make assumptions about given input images on a pixel level. Like lies pixel x,y in shadow or
	represents the pixel a part of vegetation area?
	"""
	def __init__(self, progress=None):
		BaseClass.__init__(self, progress)

	def getVegetationMaskByNDVI(
			self,
			cirData,
			threshold,
			minEntropy=0.0,
			ndviDecreaseAtLowEntropy=0.0,
			ndviIncreaseInShadow=0.0,
			ocSeSize=0,
			noPlantsMask=None,
			returnNDVI=False,
			plot=False):
		"""
		Creates mask in which vegetation will be marked.
		A CIR image will used and the NDVI will be calculated from it.
		In shadows the NDVI is lower, so the ESI will be used to find vegetation in shadows and increase the NDVI in these
		areas.
		Grass can also have a high NDVI (like on fields) but grass areas are less noisy than tree areas, therefore
		the homogeneity of a area can be calculated using the entropy. Entropy is low in homogeneity local areas and high
		in noisy ones. It is faster to calculate than homogeneity using the GLCM and the results are like the same, but
		invers. Therefore the entropy can be used to decrease the NDVI in homogeneity areas (BUT it works not that good,
		because in trees can also be homogeneity areas (in big shadows for example) and on fields, which have also high NDVI,
		the homogeneity is low because of the plants its self or from tractor lanes)

		The result will be smoothed with opening and closing. If a no plants mask is given, all marked areas in this mask
		will be removed from the vegetation mask (can be mask of streets and buildings for example).


		:param cirData: [NIR,R,G][y][x] array
		:type cirData: np.ndarray
		:param threshold: threshold for NDVI
		:type threshold: float
		:param minEntropy: threshold for min entropy in 11x11 window, if 0, no entropy calculations will be done
		:type minEntropy: float
		:param ndviDecreaseAtLowEntropy: determines how much the NDVI will be decreased if the minEntropy is not reached
		:type ndviDecreaseAtLowEntropy:
		:param ndviIncreaseInShadow: determines how much the NDVI will be increased in shadowed vegetated areas
		:type ndviIncreaseInShadow: float
		:param ocSeSize: size of structure element for opening and closing
		:type ocSeSize: int
		:param noPlantsMask: mask where no plants should be [0,255]
		:type noPlantsMask: np.ndarray | None
		:param returnNDVI: if True, a tuple with NDVI will be returned (Mask, NDVI)
		:type returnNDVI: bool
		:param plot: if True, plots will be shown
		:type plot: bool

		:return: uint8 mask [0,255] | (mask, ndvi)
		:rtype: np.ndarray | tuple
		"""
		self.setProgress(0, "Convert data to float")
		nir = cirData[0].astype(np.float16)  # type: np.ndarray
		r = cirData[1].astype(np.float16)  # type: np.ndarray
		g = cirData[2].astype(np.float16)  # type: np.ndarray
		self.setProgress(10, "Calculate NDVI")
		start = timer()
		ndvi = self.calcNDVI(nir, r)
		print "calcNDVI: %4.3f" % (timer() - start)

		if ndviIncreaseInShadow != 0:
			start = timer()
			self.setProgress(30, "Calculate SI")
			si = self.calcShadowIndex(nir, r, g, plot=plot)
			print "calcShadowIndex: %4.3f" % (timer() - start)
			start = timer()
			self.setProgress(50, "Calculate ESI")
			esi = self.calcESI(ndvi, si, clip=1.0, plot=plot)
			print "calcESI: %4.3f" % (timer() - start)
			start = timer()
			esi[esi < 0] = 0
			esi[esi > 0] = 1
			ndvi += (esi * ndviIncreaseInShadow)
			print "increaseNDVI: %4.3f" % (timer() - start)

		if minEntropy > 0 and ndviDecreaseAtLowEntropy != 0:
			self.setProgress(70, "Calculate entropy")
			entrop = entropy(ndvi, np.ones((11, 11)), plot=plot)
			ndvi -= (entrop < minEntropy) * ndviDecreaseAtLowEntropy

		self.setProgress(80, "Threshold NDVI")
		thresholdedNdvi = ndvi > threshold
		thresholdedNdvi = thresholdedNdvi.astype(np.uint8, copy=False) * 255
		if ocSeSize > 0:
			self.setProgress(90, "Opening and Closing")
			thresholdedNdvi = closing(thresholdedNdvi,
									  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ocSeSize, ocSeSize)))
			thresholdedNdvi = opening(thresholdedNdvi,
									  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ocSeSize, ocSeSize)))

		if noPlantsMask is not None:
			self.setProgress(95, "Apply NoPlantsMask")
			thresholdedNdvi = cv2.bitwise_and(thresholdedNdvi, cv2.bitwise_not(noPlantsMask))
		if plot is True:
			ParametrizedPlot(
				self.getVegetationMaskByNDVI,
				parameters=[ArrayParameter("cirData", cirData)],
				adjustableParameters=[
					AdjustableParameter("threshold", threshold, -1.0, 1.0),
					AdjustableParameter("minEntropy", minEntropy, 0.0, 8.0),
					AdjustableParameter("ndviDecreaseAtLowEntropy", ndviDecreaseAtLowEntropy, 0.0, 1.0),
					AdjustableParameter("ndviIncreaseInShadow", ndviIncreaseInShadow, 0.0, 1.0),
					AdjustableParameter("ocSeSize", ocSeSize, 0, 50),
				],
				overlayImage=ArrayParameter("cirData", cirData),
				plotNonParameters=[ArrayParameter("ndvi", ndvi)]
			)
			plt.show(block=False)
		self.setProgress(100)
		if returnNDVI is True:
			return thresholdedNdvi, ndvi
		return thresholdedNdvi

	@staticmethod
	def calcShadowIndex(r, g, b, plot=False):
		"""
		Calculates the shadow index.
		From http://elib.dlr.de/84969/1/isprsarchives-XL-1-W3-415-2013.pdf.

		:param r: red channel
		:type r: np.ndarray
		:param g: green channel
		:type g: np.ndarray
		:param b: blue channel
		:type b: np.ndarray

		:return: si in range [-1..1]
		:rtype: np.ndarray
		"""
		if r.dtype != np.float16:
			r = r.astype(np.float16)
		if g.dtype != np.float16:
			g = g.astype(np.float16)
		if b.dtype != np.float16:
			b = b.astype(np.float16)
		r /= 255
		g /= 255
		b /= 255
		r_norm = r / (r + g + b)
		setInfiniteValuesToFinite(r_norm)
		si = r_norm - r
		if plot:
			fig = plt.figure()
			fig.set_size_inches(10, 12, forward=True)
			fig.canvas.manager.set_window_title("Shadow Index (SI)")
			plt.title('si')
			plt.imshow(si, interpolation="nearest")
			plt.colorbar()
			plt.show(block=False)
		return si

	@staticmethod
	def calcNDVI(nir, r, plot=False):
		"""
		Calculates the NDVI vegetation index

		:param nir: infrared channel
		:type nir: np.ndarray
		:param r: red channel
		:type r: np.ndarray

		:return: ndvi in range [-1..1]
		:rtype: np.ndarray
		"""
		if nir.dtype != np.float16:
			nir = nir.astype(np.float16)
		if r.dtype != np.float16:
			r = r.astype(np.float16)
		ndvi = ((nir - r) / (nir + r))
		ndvi = setInfiniteValuesToFinite(ndvi)
		if plot:
			fig = plt.figure()
			fig.set_size_inches(10, 12, forward=True)
			fig.canvas.manager.set_window_title("Normalized Difference Vegetation Index (NDVI)")
			plt.title('ndvi')
			plt.imshow(ndvi, interpolation="nearest")
			plt.colorbar()
			plt.show(block=False)
		return ndvi

	def calcESI(self, ndvi, si, clip=1.0, plot=False):
		"""
		Calculates the enhanced shadow index.
		From http://elib.dlr.de/84969/1/isprsarchives-XL-1-W3-415-2013.pdf

		:param ndvi: ndvi
		:type ndvi: np.ndarray
		:param si: shadow index
		:type si: np.ndarray
		:param clip: values gets clipped at -+clip
		:type clip: float
		:return: esi in range [-clip..clip]
		:rtype: np.ndarray
		"""
		esi = ndvi / si
		esi = setInfiniteValuesToFinite(esi)
		esi[esi < -clip] = -clip
		esi[esi > clip] = clip
		esi[ndvi < 0] = -clip
		if plot:
			ParametrizedPlot(
				self.calcESI,
				parameters=[
					ArrayParameter("ndvi", ndvi, plot=True),
					ArrayParameter("si", si, plot=True)
				],
				adjustableParameters=[
					AdjustableParameter("clip", clip, 0, 100)
				],
				figTitle="Enhanced Shadow Index (ESI)"
			)
			plt.show(block=False)
		return esi

	def calcEVI(self, nir, r, b, clip=10, gain=2.5, l=1.0, c1=6.0, c2=7.5, plot=False):
		"""
		Calculates the enhanced vegetation index.
		"""
		if nir.dtype != np.float16:
			nir = nir.astype(np.float16)
		if r.dtype != np.float16:
			r = r.astype(np.float16)
		if b.dtype != np.float16:
			b = b.astype(np.float16)

		evi = gain * ((nir - r) / ((nir + c1 * r - c2 * b) + l))
		evi = setInfiniteValuesToFinite(evi)
		evi[evi > clip] = clip
		evi[evi < -clip] = -clip
		if plot:
			ParametrizedPlot(
				self.calcEVI,
				parameters=[
					ArrayParameter("nir", nir, plot=True),
					ArrayParameter("r", r),
					ArrayParameter("b", b)
				],
				adjustableParameters=[
					AdjustableParameter("gain", gain, 0, 10),
					AdjustableParameter("l", l, 0, 10),
					AdjustableParameter("c1", c1, 0, 10),
					AdjustableParameter("c2", c2, 0, 10),
					AdjustableParameter("clip", clip, 0, 100)],
				figTitle="Enhanced Vegetation Index (EVI)"
			)

			plt.show(block=False)
		return evi

	def getTreesByEsiAndNdvi(
			self,
			cirData,
			X=18,
			A=0.3,
			B=0.7,
			Y=30,
			seSize=5,
			minEntropy=0,
			plot=False):
		"""
		DEPRECATED !

		http://elib.dlr.de/84969/1/isprsarchives-XL-1-W3-415-2013.pdf
		D = ReNDVI > X
		LP = A * ReNDVI + B * ReESI + D
		IMRES = LP > Y

		:param cirData: [NIR,R,G][y][x] array
		:type cirData: np.ndarray
		:param X: NDVI values (0-25) greater X will be used
		:type X: int
		:param A: weight of NDVI
		:type A: float
		:param B: weight of ESI
		:type B: weight
		:param Y: values of LP greater Y will be used
		:type Y: int
		:param plot: if True, plots will be shown
		:type plot: bool

		:return: possible trees positions in uint8
		:rtype: np.ndarray
		"""
		nir = cirData[0].astype(np.float16, copy=False)  # type: np.ndarray
		r = cirData[1].astype(np.float16, copy=False)  # type: np.ndarray
		g = cirData[2].astype(np.float16, copy=False)  # type: np.ndarray

		start = timer()
		ndvi = self.calcNDVI(nir, r)
		end = timer()
		print "NDVI " + str(end - start)
		# shadow index using nir
		start = timer()
		si = self.calcShadowIndex(nir, r, g)
		end = timer()
		print "SI " + str(end - start)
		# si = removeOutliers(si, m=6., median=3)


		# enhanced shadow index
		start = timer()
		esi = self.calcESI(ndvi, si)
		end = timer()
		print "ESI " + str(end - start)
		# esi = removeOutliers(esi, m=99., median=1, plot=False)
		# ndvi = removeOutliers(ndvi, m=6., median=1, plot=False)

		start = timer()
		# bring to 0..255
		esi = scaleToUInt(esi, copy=False)
		ndvi = scaleToUInt(ndvi, copy=False)
		# classify in 25 classes with equal intervals (10)
		esiReclassified = reclassify(esi, 25)
		ndviReclassified = reclassify(ndvi, 25)
		end = timer()
		print "Reclassify " + str(end - start)
		# linear production
		# D = ReNDVI > X
		# LP = A * ReNDVI + B * ReESI + D
		# IMRES = LP > Y
		start = timer()
		D = np.zeros(ndviReclassified.shape, np.uint8)
		D[ndviReclassified > X] = ndviReclassified[ndviReclassified > X]
		LP = A * ndviReclassified + B * esiReclassified + D
		IMRES = LP > Y
		if minEntropy > 0:
			entrop = entropy(ndvi, np.ones((11, 11)), plot=plot)
			IMRES = (IMRES == True) & (entrop > minEntropy)
		IMRES = scaleToUInt(IMRES.astype(np.uint8, copy=False), copy=False)
		end = timer()
		print "FUNCTION " + str(end - start)
		# linear production

		if seSize > 0:
			start = timer()
			IMRES = closing(IMRES, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (seSize, seSize)))
			IMRES = opening(IMRES, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (seSize, seSize)))
			end = timer()
			print "MORPH " + str(end - start)

		if plot is True:
			plt.figure().set_size_inches(10, 12, forward=True)
			plt.subplot(211)
			plt.title('cir image')
			plt.imshow(cirData.transpose((1, 2, 0)), interpolation="nearest")
			plt.colorbar()
			plt.subplot(212)
			plt.title('ndvi')
			plt.imshow(ndvi, interpolation="nearest")
			plt.colorbar()

			plt.figure().set_size_inches(10, 12, forward=True)
			plt.subplot(211)
			plt.title('si')
			plt.imshow(si, interpolation="nearest")
			plt.colorbar()
			plt.subplot(212)
			plt.title('esi (ndvi/si) w/o outliers')
			plt.imshow(esi, interpolation="nearest")
			plt.colorbar()

			plt.figure().set_size_inches(10, 12, forward=True)
			plt.subplot(211)
			plt.title('LP')
			plt.imshow(LP, interpolation="nearest")
			plt.colorbar()
			plt.subplot(212)
			plt.title('LP>Y')
			plt.imshow(IMRES, interpolation="nearest")
			plt.colorbar()

			ParametrizedPlot(
				self.getTreesByEsiAndNdvi,
				parameters=[ArrayParameter("cirData", cirData)],
				adjustableParameters=[
					AdjustableParameter("X", X, 0, 25),
					AdjustableParameter("A", A, 0, 1),
					AdjustableParameter("B", B, 0, 1),
					AdjustableParameter("Y", Y, 0, 100),
					AdjustableParameter("seSize", seSize, 0, 50)
				],
				overlayImage=ArrayParameter("cirData", cirData)
			)
			plt.show(block=False)
		return IMRES


class PlantParameters(BaseClass):
	def __init__(
			self,
			geoTransform,
			pixelPerMeter=5,
			demFilePath="",
			minPlantRadius=0.25,
			maxBushRadius=1,
			maxTreeRadius=8,
			bushClassesCount=1,
			treeClassesCount=1,
			featureVectorDataList=None,
			forestMaskPath="",
			progress=None
	):
		"""
		Used for extraction information about single plants from given plant mask.
		At the end a vector file with needed info about each plant will be created.

		:param geoTransform: gdal geoTransform
		:type geoTransform: tuple
		:param pixelPerMeter:
		:type pixelPerMeter: int | float
		:param demFilePath: dem file path for z value extraction
		:type demFilePath: str
		:param minPlantRadius: plants smaller this radius will be removed
		:type minPlantRadius: float | int
		:param maxTreeRadius: plants bigger this radius will be splitted up in smaller ones
		:type maxTreeRadius: float | int
		:param featureVectorDataList: list of gray scale image arrays (same size as mask)
		:type featureVectorDataList: list[FeatureVectorData] | None
		:param forestMaskPath: path of forestMask, no bushes will be created in forests, because they will be made as grass
		:type forestMaskPath: str
		:type progress: function
		"""
		BaseClass.__init__(self, progress)

		self.geoTransform = geoTransform
		self.minPlantRadius = minPlantRadius
		self.maxBushRadius = maxBushRadius
		self.maxTreeRadius = maxTreeRadius
		self.demFilePath = demFilePath
		self._demData = None
		self.demGeoTransform = None
		self.pixelPerMeter = pixelPerMeter
		self.bushClassesCount = bushClassesCount
		self.treeClassesCount = treeClassesCount
		self.featureVectorDataList = featureVectorDataList
		self.forestMaskPath = forestMaskPath
		self._forestMask = None
		self.tooSmallPlants = 0  # counter for too small plants (for information purposes)

	@property
	def demData(self):
		# open dem for z value extraction
		if self._demData is None and self.demFilePath != "":
			demDataSet = gdal.Open(self.demFilePath, gdal.GA_ReadOnly)  # type: gdal.Dataset
			self.demGeoTransform = demDataSet.GetGeoTransform()
			self._demData = np.array(demDataSet.ReadAsArray(), dtype=np.uint16)
			demDataSet = None
		return self._demData

	@property
	def hasDemData(self):
		return self.demData is not None

	@property
	def forestMask(self):
		# open forest mask to make classification for differing between Broadleaved and Coniferous trees.
		if self._forestMask is None and self.forestMaskPath != "":
			dataSet = gdal.Open(self.forestMaskPath, gdal.GA_ReadOnly)  # type: gdal.Dataset
			self._forestMask = np.array(dataSet.ReadAsArray(), dtype=np.uint8)
			dataSet = None
		return self._forestMask

	@forestMask.setter
	def forestMask(self, value):
		"""
		:type value: np.ndarray
		"""
		self._forestMask = value

	@property
	def hasForestMask(self):
		return self.forestMask is not None

	def extractFromMask(
			self,  # type: PlantParameters
			savePath,
			plantMask,
			noPlantsMask=None,
			minDistance=9,
			seType=None,
			useRadiusForClassification=False,
			plot=False):

		"""
		Extracts plant position and size from given plantMask using watershed transform and subdividing too big plants.
		Plants will also be classified by a given feature vector.

		:param savePath: .shp path to save parameter
		:type savePath: str
		:param plantMask: mask where plants are
		:type plantMask: np.ndarray
		:param noPlantsMask: mask where no plants will be (i.e. street/building mask)
		:type noPlantsMask: np.ndarray | None
		:param minDistance: half size of local maxima search in pixel
		:type minDistance: int
		:param seType: opencv type of structure element i.e. cv2.MORPH_RECT for maxima search
		:type seType: int | None
		:param useRadiusForClassification: If True, the radius will also be used for classification.
		:type useRadiusForClassification: bool
		:param plot: shows plots
		:type plot: bool


		:rtype: list[PlantClassStats]
		"""
		self.tooSmallPlants = 0  # just for information purpose

		if plantMask.dtype != np.uint8:
			plantMask = plantMask.astype(np.uint8)

		# split mask in middle of streets
		# if noPlantsMask is not None:
		# 	self.setProgress(0, "Skeletonize mask")
		# 	start = timer()
		# 	streetBuildingMaskSkeleton = skimage.morphology.skeletonize(noPlantsMask / 255)
		# 	plantMask[streetBuildingMaskSkeleton == True] = 0
		# 	print "skeletonize: %4.3f" % (timer() - start)

		self.setProgress(5, "Watershed transform")
		markers, labelsCount, labels, stats, centroids = self.getLabelsWithStatsByWatershededDistanceTransform(
			binImage=plantMask,
			minDistance=minDistance,
			seType=seType,
			noMarkerMask=noPlantsMask,
			plot=plot
		)
		self.setProgress(30, "Add plants to output file")
		# create output file
		driver = ogr.GetDriverByName("ESRI Shapefile")
		srs = osr.SpatialReference()
		srs.ImportFromEPSG(25832)
		outDataSource = driver.CreateDataSource(savePath)  # type: ogr.DataSource
		layer = outDataSource.CreateLayer("plants", srs=srs, geom_type=ogr.wkbPoint)  # type: ogr.Layer

		# create a field for radius
		fieldName = "radius"
		fieldDefn = ogr.FieldDefn(fieldName, ogr.OFTReal)
		layer.CreateField(fieldDefn)

		# create a field for z values
		fieldName = "z"
		fieldDefn = ogr.FieldDefn(fieldName, ogr.OFTInteger)
		layer.CreateField(fieldDefn)

		# create a field for type (tree or bush)
		fieldName = "type"
		fieldDefn = ogr.FieldDefn(fieldName, ogr.OFTString)
		layer.CreateField(fieldDefn)

		# create a field for feature values
		if self.featureVectorDataList is not None:
			for featureVector in self.featureVectorDataList:
				fieldName = featureVector.name
				fieldDefn = ogr.FieldDefn(fieldName, ogr.OFTReal)
				layer.CreateField(fieldDefn)

		fieldName = "class"
		fieldDefn = ogr.FieldDefn(fieldName, ogr.OFTInteger)
		layer.CreateField(fieldDefn)

		self.addPlantsToLayer(
			layer=layer,
			markers=markers,
			labels=labels,
			stats=stats,
			centroids=centroids,
			maxRadius=self.maxTreeRadius,
			noPlantsMask=noPlantsMask
		)
		self.setProgress(90, "Classify plants")
		if useRadiusForClassification:
			if self.featureVectorDataList is None:
				self.featureVectorDataList = []
			self.featureVectorDataList.append(FeatureVectorData("radius", value=None, plantType="all"))

		plantClassesStats = []
		PlantClassStats.uniqueIds = 0  # start index always from 0 per export
		plantClassesStats.extend(self.classifyPlants(
			layer,
			classesCount=self.bushClassesCount,
			plantType="bush",
			featureVectorDataList=self.featureVectorDataList
		))
		plantClassesStats.extend(self.classifyPlants(
			layer,
			classesCount=self.treeClassesCount,
			plantType="tree",
			featureVectorDataList=self.featureVectorDataList
		))

		self.setProgress(100)
		print "amount of too small plants: %d" % self.tooSmallPlants
		outDataSource.SyncToDisk()
		outDataSource = None
		return plantClassesStats

	def getLabelsWithStatsByWatershededDistanceTransform(
			self,  # type: PlantParameters
			binImage,
			minDistance=9,
			seType=None,
			noMarkerMask=None,
			plot=False
	):
		"""
		Does a distance transform on binImage, searches for local maximas in distance image using minDistance and seType,
		uses the maximas as markers for a watershed transform.


		:param binImage: binary image (i.e. mask of plants)
		:type binImage: np.ndarray
		:param minDistance: minimal distance for local maximas
		:type minDistance: int
		:param seType: structure element for local maxima search
		:type seType: int
		:param noMarkerMask: mask where no markers will be
		:type noMarkerMask: np.ndarray | None
		:param plot: shows plots
		:type plot: bool

		:return: markers, labelsCount, labels, stats, centroids
		:rtype: tuple
		"""
		start = timer()
		distance = cv2.distanceTransform(binImage, cv2.DIST_L2, 5)  # type: np.ndarray
		localMaxiImage = getLocalMaxima(distance, minDistance, seType)
		# remove peaks which are on noMarkerMask (i.e.street/building-mask)
		if noMarkerMask is not None:
			# localMaxiImage &= ~noMarkerMask
			localMaxiImage = cv2.bitwise_and(localMaxiImage, cv2.bitwise_not(noMarkerMask))

		markersCount, markers = cv2.connectedComponents(localMaxiImage, connectivity=8)
		labelsCount, labels, stats, centroids = self.getLabelsAndStatsByWatershedCV(binImage, markers)

		print "getLabelsWithStatsByWatershededDistanceTransform: %4.3f" % (timer() - start)
		if plot:
			fig = plt.figure()
			fig.set_size_inches(10, 12, forward=True)
			fig.canvas.manager.set_window_title("getLabelsWithStatsByWatershededDistanceTransform")
			ax = plt.subplot(311)
			plt.title('binImage')
			plt.imshow(binImage, interpolation="nearest", cmap="gray")
			plt.colorbar()
			ax = plt.subplot(312)
			plt.title('distance')
			plt.imshow(distance, interpolation="nearest", cmap="gray")
			plt.colorbar()
			ax.autoscale(False)
			plt.plot(np.where(localMaxiImage > 0)[1], np.where(localMaxiImage > 0)[0], ".r")  #
			ax = plt.subplot(313)
			plt.title('splitted labels')
			plt.imshow(labels, interpolation="nearest")
			plt.colorbar()

			plt.show(block=False)
		return markers, labelsCount, labels, stats, centroids

	@staticmethod
	def getRandomPointsMarkers(image, borderSize=11, maxTreeRadius=30, seed=0):
		"""
		Creates random labeled positions on a given image.

		:param image: binary image
		:type image: np.ndarray
		:param borderSize: Border of image where no points will be created. (size of erosion kernel)
		:type borderSize: int
		:param maxTreeRadius: maximal tree radius in pixel
		:type maxTreeRadius: int
		:param seed: seed of the random vars
		:type seed: int

		:return: labeled random positions
		:rtype: np.ndarray
		"""
		# create grid positions
		height, width = image.shape
		plantsInX = np.linspace(borderSize, width - borderSize, max(int(width / (maxTreeRadius * 2)), 3))
		plantsInY = np.linspace(borderSize, height - borderSize, max(int(height / (maxTreeRadius * 2)), 3))
		plantsInX, plantsInY = np.meshgrid(plantsInX, plantsInY)
		plantsInX, plantsInY = plantsInX.ravel(), plantsInY.ravel()
		# randomize grid positions
		np.random.seed(seed)
		maxPositionOffset = maxTreeRadius * 0.6
		rx = plantsInX + np.random.uniform(low=-maxPositionOffset, high=maxPositionOffset, size=plantsInX.shape)
		ry = plantsInY + np.random.uniform(low=-maxPositionOffset, high=maxPositionOffset, size=plantsInY.shape)
		rx = rx.astype(np.uint32, copy=False)
		ry = ry.astype(np.uint32, copy=False)
		rx[rx >= width] = width - 1
		ry[ry >= height] = height - 1
		positionsMask = np.zeros(image.shape, np.uint8)
		positionsMask[ry, rx] = 1
		# remove points which are not in eroded image mask
		positionsMask = cv2.bitwise_and(positionsMask,
										cv2.erode(image, np.ones((borderSize, borderSize), np.uint8), borderValue=0))
		count, markers = cv2.connectedComponents(positionsMask, connectivity=8)
		return count, markers

	@staticmethod
	def getLabelsAndStatsByWatershedCV(image, markers):
		"""
		Watershed using opencv. Note: image can only be uint8!

		:param image: uint8 image
		:type image: np.ndarray
		:param markers:
		:type markers: np.ndarray

		:return: count, labels, stats, centroids from cv2.connectedComponentsWithStats()
		:rtype: tuple
		"""
		assert image.dtype == np.uint8, "Wrong data type"
		labelsWatershed = markers.copy()
		cv2.watershed(np.array([image, image, image], np.uint8).transpose(1, 2, 0), labelsWatershed)
		# opencv labels background, so remove him and relabel with connected components
		# (opencv's watershed draws borders between labels)
		# This is still faster than skimage's watershed, but what about memory footprint?
		labelsWatershed[image == 0] = 0
		labelsWatershed = (labelsWatershed > 0).astype(np.uint8, copy=False)
		count, labels, stats, centroids = cv2.connectedComponentsWithStats(labelsWatershed, connectivity=4)
		return count, labels, stats, centroids

	def addPlantsToLayer(
			self,  # type: PlantParameters
			layer,
			markers,
			labels,
			stats,
			centroids,
			maxRadius,
			previousCutoutX=0,
			previousCutoutY=0,
			noPlantsMask=None,
			plot=False
	):
		"""
		Extracts plant radius by the area of a label and writes data to given shape file layer.
		If radius is too big, the area will recursively be split up by random points.

		To enable a later classification,
		For each image in the feature vector, the mean of all pixels of a labeled area will be saved to attribute.

		With the given forest mask no bushes will be created in forests.

		:param layer: layer where plants will be added
		:type layer: ogr.Layer
		:param labels: labels from cv2.connectedComponentsWithStats()
		:type labels: np.ndarray
		:param markers: markers of the labels (needed if centroid is on noPlantsMask, the marker will be used)
		:type markers: np.ndarray
		:param stats: stats from cv2.connectedComponentsWithStats()
		:type stats: np.ndarray
		:param centroids: centroids from cv2.connectedComponentsWithStats()
		:type centroids: np.ndarray
		:param maxRadius: maximal plant radius
		:type maxRadius: float | int
		:param previousCutoutX: the pixels position x of the previous cutout operation (needed for recursive call)
		:type previousCutoutX: int
		:param previousCutoutY: the pixels position y of the previous cutout operation (needed for recursive call)
		:type previousCutoutY: int
		:param noPlantsMask: mask where no plants should be
		:type noPlantsMask:  np.ndarray
		:param plot:
		:type plot: bool
		"""
		labelsMax = labels.max()
		for i in range(1, labelsMax + 1):
			x, y, width, height, area = stats[i]
			r = math.sqrt(area / math.pi) / self.pixelPerMeter  # assumption: trees/bushes are round
			if r > maxRadius:  # if radius is too big, split the area by random points, else add plant to layer

				# extract current label
				objectCutout = labels[y:y + height, x:x + width]
				objectMask = np.ones(objectCutout.shape, dtype=np.uint8)
				objectMask[objectCutout != i] = 0

				# distribute points on obj
				randomMarkersCount = 0
				seed = 0
				maxRadiusInPixel = maxRadius * self.pixelPerMeter
				abort = False
				while randomMarkersCount - 1 < 2:  # find at least 2 two markers (background is also counted, so -1)
					randomMarkersCount, randomMarkers = self.getRandomPointsMarkers(
						image=objectMask,
						borderSize=int(maxRadiusInPixel / 2) * 2 + 1,
						maxTreeRadius=maxRadiusInPixel,
						seed=seed
					)
					seed += 1
					if maxRadiusInPixel > 3:
						maxRadiusInPixel -= max(int(maxRadiusInPixel * 0.1), 1)  # 10% smaller per try
					else:
						# abort if no 2 markers are found after some iterations
						abort = True
						coordX, coordY = pixel2coord(self.geoTransform, centroids[i][0], centroids[i][1])
						print "abort"
						# print "abort finding random points on label after %d iterations. radius: %3.2f m, width: %d px," \
						# 	  " height: %d px, coordX: %f4.4, coordY %f4.4  " % (seed, r, width, height, coordX, coordY)
						break
				if abort is True:
					continue
				randomCount, randomLabels, randomStats, randomCentroids = self.getLabelsAndStatsByWatershedCV(
					objectMask, randomMarkers)
				# add cutout offset from and offset from previous cutouts
				offsetX = x + previousCutoutX
				offsetY = y + previousCutoutY
				randomCentroids[:, 0] += offsetX
				randomCentroids[:, 1] += offsetY
				self.addPlantsToLayer(
					layer=layer,
					markers=randomMarkers,
					labels=randomLabels,
					stats=randomStats,
					centroids=randomCentroids,
					maxRadius=maxRadius,
					noPlantsMask=noPlantsMask,
					previousCutoutX=offsetX,
					previousCutoutY=offsetY,
					plot=plot
				)
			else:
				self.setProgress(i / float(labelsMax) * 100, "Add plants to output file %d / %d" % (i, labelsMax))
				if r > self.minPlantRadius:
					centerX = centroids[i][0]
					centerY = centroids[i][1]
					if noPlantsMask is not None:
						isCenterOnNoPlantsMask = noPlantsMask[int(centerY), int(centerX)] > 0
						if isCenterOnNoPlantsMask:  # if centroid is on street, use marker position instead
							markersCutout = markers[y:y + height, x:x + width]
							markerPos = np.where(markersCutout == i)
							if markerPos[0].size > 0:
								centerX, centerY = markerPos[1][0] + previousCutoutX, markerPos[0][0] + previousCutoutY

					# add data to output layer
					featureDefn = layer.GetLayerDefn()
					centerXCoord, centerYCoord = pixel2coord(self.geoTransform, centerX, centerY)
					treeFeature = ogr.Feature(featureDefn)

					# extract z value
					if self.hasDemData:
						demCenterX, demCenterY = coord2pixel(self.demGeoTransform, centerXCoord, centerYCoord)
						if demCenterX < self.demData.shape[1] and demCenterY < self.demData.shape[0]:
							treeFeature.SetField2("z", self.demData[demCenterY][demCenterX])
						else:
							treeFeature.SetField("z", 0)
					else:
						treeFeature.SetField("z", 0)
					treeFeature.SetField("radius", r)

					# determine if it is a bush or tree
					# no bushes in forest (because they will be made by grass mask)
					if self.hasForestMask:
						isInForest = self.forestMask[int(centerY), int(centerX)] > 0
						if isInForest:
							treeFeature.SetField("type", "tree")
						else:
							if r <= self.maxBushRadius:
								treeFeature.SetField("type", "bush")
							else:
								treeFeature.SetField("type", "tree")
					else:
						if r <= self.maxBushRadius:
							treeFeature.SetField("type", "bush")
						else:
							treeFeature.SetField("type", "tree")

					treeFeature.SetField("class", 0)  # default class

					# extract features
					if self.featureVectorDataList is not None:
						for featureVectorData in self.featureVectorDataList:
							if featureVectorData.value is not None:
								image = featureVectorData.value
								imageCutout = image[previousCutoutY + y:previousCutoutY + y + height,
											  previousCutoutX + x:previousCutoutX + x + width]

								# extract current label
								objectCutout = labels[y:y + height, x:x + width]
								objectMask = np.ones(objectCutout.shape, dtype=np.uint8)
								objectMask[objectCutout != i] = 0

								# calc mean of all pixels under current label
								mean = cv2.mean(imageCutout, objectMask)[0]
								treeFeature.SetField(featureVectorData.name, mean)

					pointGeometry = ogr.Geometry(ogr.wkbPoint)
					pointGeometry.SetPoint_2D(0, centerXCoord, centerYCoord)
					treeFeature.SetGeometry(pointGeometry)
					layer.CreateFeature(treeFeature)
				else:
					self.tooSmallPlants += 1

	def classifyPlants(
			self,
			layer,
			classesCount,
			plantType="tree",
			featureVectorDataList=None
	):

		"""
		Creates classes using kmeans algorithm.

		Each plant can have information of featuresVector in attributes. With this data a feature vector will be
		created for the kmeans algorithm. The algorithm will find centers of the data and cluster all data to this centers.
		I.e. we want 2 classes and have 6 plants with radius of [1,2,3,4,5,6]. The algorith will find the 2 and 5 as centers
		and 1,2,3 will be the first class because they are near center of 2 and 4,5,6 will be the second class.

		:param layer:
		:type layer: ogr.Layer
		:param classesCount: number of classes to be generated
		:type classesCount: int
		:param plantType: which plant type should be used for classification "bush" or "tree"
		:type plantType: str
		:param featureVectorDataList: list of feature vector data
		:type featureVectorDataList: list[FeatureVectorData] | None

		:return: info about generated classes
		:rtype: list[PlantClassStats]
		"""
		layer.SetAttributeFilter("type='%s'" % plantType)
		plantCount = layer.GetFeatureCount()
		featuresCount = 0
		# get only features for this plant type
		if featureVectorDataList is not None:
			featureVectorDataListForPlantType = []
			for featureVectorData in featureVectorDataList:
				if featureVectorData.plantType == plantType or featureVectorData.plantType == "all":
					featureVectorDataListForPlantType.append(featureVectorData)

			featuresCount = len(featureVectorDataListForPlantType)

		if featuresCount == 0:
			print "There must be at least one feature to make classification"
			return [PlantClassStats(type=plantType, count=plantCount, classificationData=[])]
		if plantCount < classesCount:
			print "Too few plants (plantsCount<classesCount) to make classification"
			return [PlantClassStats(type=plantType, count=plantCount, classificationData=[])]

		featureVector = np.zeros((plantCount, featuresCount), dtype=np.float64)
		layer.ResetReading()
		# extract featureVector
		for i in range(0, plantCount):
			feature = layer.GetNextFeature()  # type: ogr.Feature
			for j in range(0, len(featureVectorDataListForPlantType)):
				featureVector[i][j] = feature.GetFieldAsDouble(featureVectorDataListForPlantType[j].name)

		# k means clustering
		kmeans = KMeans(n_clusters=classesCount, random_state=0)
		kmeans.fit(featureVector)

		layer.ResetReading()
		for i in range(0, plantCount):
			feature = layer.GetNextFeature()  # type: ogr.Feature
			feature.SetField("class", kmeans.labels_[i] + PlantClassStats.uniqueIds)
			layer.SetFeature(feature)  # saves changes

		stats = self.getClassStats(
			layer=layer,
			plantType=plantType,
			featureVectorDataList=featureVectorDataListForPlantType,
			clusterCenters=kmeans.cluster_centers_
		)
		layer.SetAttributeFilter("")
		return stats

	@staticmethod
	def getClassStats(layer, plantType, featureVectorDataList, clusterCenters):
		"""
		Returns statistics of plant type like plant count and cluster centers.

		:param layer:
		:type layer: ogr.Layer
		:param plantType: "bush" or "tree"
		:type plantType: str
		:param featureVectorDataList:
		:type featureVectorDataList: list[FeatureVectorData]
		:param clusterCenters: cluster centers of feature vector
		:type clusterCenters: list

		:return:
		:rtype: list[PlantClassStats]
		"""
		stats = []
		for i in range(0, len(clusterCenters)):
			plantStats = PlantClassStats(
				type=plantType,
				count=0,
				classificationData=[]
			)
			layer.SetAttributeFilter("type='%s' AND class=%d" % (plantType, plantStats.classId))
			plantStats.count = layer.GetFeatureCount()
			for j in range(0, len(clusterCenters[i])):
				plantStats.classificationData.append(
					{
						featureVectorDataList[j].name: clusterCenters[i][j]
					}
				)
			stats.append(plantStats)
		return stats

	def getPlantsAtCadastre(self, savePath, cadastrePolygonsPath, plantsMaskPath, distanceBetweenPoints=1, threshold=1):
		"""
		Creates plants at cadastre borders.

		Note: createPointsAlongLines uses qgis functionality, so there is dependency to qgis.

		:param savePath:
		:param cadastrePolygonsPath:
		:param plantsMaskPath:
		:param distanceBetweenPoints:
		:param threshold:
		"""
		filePathWithoutExtension, fileExtension = os.path.splitext(savePath)

		# Shrink to avoid overlapping borders
		polygonsBufferedPath = filePathWithoutExtension + "_bufferedPolygons.shp"
		createBuffers(
			inputPath=cadastrePolygonsPath,
			outputPath=polygonsBufferedPath,
			buffer=-0.5,
			outputFormat="ESRI Shapefile"
		)

		# Convert polygons to lines
		linesPath = filePathWithoutExtension + "_lines.shp"
		convertPolygonsToLines(polygonsBufferedPath, linesPath)

		# Creating points along lines
		# todo: dont use qgis functionality here
		pointsPath = filePathWithoutExtension + "_points.shp"
		createPointsAlongLines(
			inputPath=linesPath,
			outputPath=pointsPath,
			distanceBetweenPoints=distanceBetweenPoints
		)

		# Removing points where no vegetation is
		rasterDataSet = gdal.Open(plantsMaskPath)
		rasterData = np.array(rasterDataSet.ReadAsArray(), dtype=np.float16)
		rasterGeoTransform = rasterDataSet.GetGeoTransform()
		rasterDataSet = None
		getPointsGreaterThanThresholdInRasterImage(
			outputPath=savePath,
			vectorPath=pointsPath,
			rasterData=rasterData,
			rasterGeoTransform=rasterGeoTransform,
			pixelRadius=5,
			threshold=threshold
		)

		# clean up
		deleteFile(polygonsBufferedPath)
		# deleteFile(linesPath)
		deleteFile(pointsPath)

	def addCadastrePlantsToPlantsParameters(self, plantParametersPath, cadastrePlantsPath, radius, plantClassesStats=None):
		"""
		Adds the cadastre plants to the already created other plants.

		:param plantParametersPath:
		:type plantParametersPath: str
		:param cadastrePlantsPath:
		:type cadastrePlantsPath: str
		:param radius: radius of plants at cadastre border
		:type radius: float | str
		:param plantClassesStats: stats of plant classes in plantParametersPath
		:type plantClassesStats: list[PlantClassStats]
		"""
		parameterPlantsDataSource = ogr.Open(plantParametersPath, gdal.GA_Update)  # type: ogr.DataSource
		parameterPlantsLayer = parameterPlantsDataSource.GetLayer()  # type: ogr.Layer

		cadastrePlantsDataSource = ogr.Open(cadastrePlantsPath)  # type: ogr.DataSource
		cadastrePlantsLayer = cadastrePlantsDataSource.GetLayer()  # type: ogr.Layer
		cadastrePlantClass = PlantClassStats("bush", cadastrePlantsLayer.GetFeatureCount(), [{"radius": radius}, {"isHedge": 1}])
		if plantClassesStats is None:
			plantClassesStats = []
		plantClassesStats.append(cadastrePlantClass)
		for cadastrePlant in cadastrePlantsLayer:  # type: ogr.Feature
			cadastrePlantGeom = cadastrePlant.geometry()  # type: ogr.Geometry
			coordX, coordY = cadastrePlantGeom.GetPoint_2D()
			outputFeature = ogr.Feature(parameterPlantsLayer.GetLayerDefn())
			# z pos
			if self.hasDemData:
				demCenterX, demCenterY = coord2pixel(self.demGeoTransform, coordX, coordY)
				if demCenterX < self.demData.shape[1] and demCenterY < self.demData.shape[0]:
					outputFeature.SetField2("z", self.demData[demCenterY][demCenterX])
				else:
					outputFeature.SetField("z", 0)
			else:
				outputFeature.SetField("z", 0)
			outputFeature.SetField("radius", radius)
			outputFeature.SetField("type", cadastrePlantClass.type)
			outputFeature.SetField("class", cadastrePlantClass.classId)
			outputFeature.SetGeometry(cadastrePlantGeom)
			parameterPlantsLayer.CreateFeature(outputFeature)

		parameterPlantsDataSource = None
		cadastrePlantsDataSource = None
		return plantClassesStats


class PlantClassStats:
	"""
	Holds information about a plant class.
	"""
	uniqueIds = 0

	def __init__(self, type, count, classificationData):
		"""
		:param type: "tree" or "bush"
		:type type: str
		:param count: number of plants
		:type count: int
		:param classificationData: list of dict with a name and value ie. [{"radius": 0.9}]
		:type classificationData: list[dict]
		"""
		self._id = PlantClassStats.uniqueIds
		PlantClassStats.uniqueIds += 1
		self.type = type
		self.count = count
		self.classificationData = classificationData

	def __str__(self):
		classDataStr = ""
		for classData in self.classificationData:
			classDataStr += str(classData)
		return "ClassId: %d, Type: %s, Count: %d, ClassCenters: %s" % (
		self.classId, self.type, self.count, classDataStr)

	@property
	def classId(self):
		return self._id


class FeatureVectorData:
	"""
	Holds data for unsupervised classification.
	"""
	def __init__(self, name, value, plantType):
		"""


		:param name: name of feature. MAX LENGTH is 10 !!! (because of shape file column name length limitation)
		:type name: str
		:param value: data of feature vector, can be None, but then only the name will be used (i.e. for radius).
		If value is a path it will be opened and converted to np.ndarray.
		:type value: np.ndarray | None | str

		:param plantType: determines for which plant type this feature will be used for classification (ie. "tree" or "bush" or "all")
		:type plantType: str

		"""
		self.name = name[:10]  # 10 is max length of column name in shape files
		if len(name) > 10:
			print "FeatureVectorData name too long. It will be cut from '%s' to '%s'" % (name, self.name)
		if isinstance(value, str) or isinstance(value, unicode):
			dataSet = gdal.Open(value, gdal.GA_ReadOnly)  # type: gdal.Dataset
			if dataSet is not None:
				self.value = np.array(dataSet.ReadAsArray(), dtype=np.uint8)
			else:
				self.value = value
			dataSet = None
		else:
			self.value = value
		self.plantType = plantType

	def __str__(self):
		return "Name: %s, Value: %s, PlantType: %s" % (self.name, self.value, self.plantType)
