import os
import subprocess
import sys

import gdal
import numpy as np
from osgeo import ogr

from files import deleteFile
from base_class import getSubprocessStartUpInfo
from collections import Counter

from raster import coord2pixel

import cv2

import subprocess
from gdalconst import GA_ReadOnly, GA_Update
from PlantPlanter.utils.files import deleteFile



def clipping(inputMask, inputRaster, clippedPathTemp):
	"""
	Clips a given input raster (e.g. DOP) using a raster mask
	"""

	inputRaster = cv2.imread(inputRaster)
	streetMaskPath = cv2.imread(inputMask, 0)
	clipped = cv2.bitwise_and(inputRaster, inputRaster, mask=streetMaskPath)
	cv2.imwrite(clippedPathTemp, clipped)

def CopyExtentandEPSG(EEPSG, noEEPSG, targetPath):
	"""
	Copy extent and EPSG from input raster to output raster
	"""

	i = 0
	while i < 2:
		source_extent = gdal.Open(EEPSG, GA_ReadOnly)
		target_extent = gdal.Open(noEEPSG, GA_Update)

		target_extent.SetGeoTransform(source_extent.GetGeoTransform())
		i = i + 1

	subprocess.call('gdalwarp ' + noEEPSG + ' ' + targetPath + ' -t_srs "+proj=utm +zone=32 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs"')

	# cleanup
	target_extent = None
	deleteFile(noEEPSG)


def vector2raster(extent, inputPath, outputPath, pixelSize=0.2):
	"""
	Converts a given input polygon vector layer to a raster image.


	:param extent: (minX, minY, maxX, maxY)
	:type extent: tuple
	:param inputPath: path to polygon vector layer
	:type inputPath: str
	:param outputPath: path where raster will be saved
	:type outputPath: str
	:param pixelSize: size of pixel in georeferenced units (ie. meter) of input data
	:type pixelSize: float
	"""
	subprocess.call(
		[
			"gdal_rasterize",
			"-of", "GTiff",
			"-ot", "Byte",
			"-init", "0",
			"-burn", "255",
			"-te", str(extent[0]), str(extent[1]), str(extent[2]), str(extent[3]),
			"-tr", str(pixelSize), str(pixelSize),
			inputPath,
			outputPath
		],
		startupinfo=getSubprocessStartUpInfo()
	)


def convertToEPSG(inputPath, outputPath, outputFormat="GeoJSON", toEPSG=25832):
	"""
	Converts the given input path to the given EPSG and saves it in outputPath.

	:param inputPath:
	:type inputPath: str
	:param outputPath:
	:type outputPath: str
	:param outputFormat: ogr2ogr output format http://www.gdal.org/ogr2ogr.html
	:type outputFormat: str
	:param toEPSG:
	:type toEPSG: int
	"""
	deleteFile(outputPath)
	subprocess.call(
		[
			"ogr2ogr",
			"-f", outputFormat,
			"-t_srs", "EPSG:%d" % toEPSG,
			outputPath,
			inputPath
		],
		startupinfo=getSubprocessStartUpInfo()
	)


def convertLinesToPolygons(inputPath, outputPath):
	"""
	Converts lines of inputPath file to polygons and saves at outputPath.
	Fields and Attributes will also be saved.

	:type inputPath: str
	:type outputPath: str
	"""
	inputDataSource = ogr.Open(inputPath)  # type: ogr.DataSource
	inputLayer = inputDataSource.GetLayerByIndex(0)  # type: ogr.Layer

	# create output shape file
	deleteFile(outputPath)
	outputDriver = inputDataSource.GetDriver()
	outputDataSource = outputDriver.CreateDataSource(outputPath)  # type: ogr.DataSource

	outputLayer = outputDataSource.CreateLayer(
		"polygons",
		srs=inputLayer.GetSpatialRef(),
		geom_type=ogr.wkbPolygon)

	# create attribute fields
	firstFeature = inputLayer.GetNextFeature()  # read first feature to get attributes
	if firstFeature is not None:
		for i in range(firstFeature.GetFieldCount()):
			fieldDef = firstFeature.GetFieldDefnRef(i)
			outputLayer.CreateField(fieldDef)

		inputLayer.ResetReading()
		for feature in inputLayer:  # type: ogr.Feature
			# copy attributes
			outputFeature = ogr.Feature(inputLayer.GetLayerDefn())
			for fieldId in range(0, feature.GetFieldCount()):
				outputFeature.SetField2(fieldId, feature.GetField(fieldId))
			# create geometry
			geom = feature.geometry()  # type: ogr.Geometry
			if geom.IsRing():
				outputGeom = ogr.Geometry(ogr.wkbPolygon)
				ring = ogr.Geometry(ogr.wkbLinearRing)
				for i in range(0, geom.GetPointCount()):
					x, y = geom.GetPoint_2D(i)
					ring.AddPoint_2D(x, y)
				outputGeom.AddGeometry(ring)
				outputFeature.SetGeometry(outputGeom)
				outputLayer.CreateFeature(outputFeature)

	# save and cleanup
	outputDataSource.SyncToDisk()
	outputDataSource = None
	inputDataSource = None


def convertPolygonsToLines(inputPath, outputPath):
	"""
	Converts polygons of inputPath file to lines and saves at outputPath.
	Fields and Attributes will also be saved.
	Note: No multipolygon support right now.

	:type inputPath: str
	:type outputPath: str
	"""
	inputDataSource = ogr.Open(inputPath)  # type: ogr.DataSource
	inputLayer = inputDataSource.GetLayerByIndex(0)  # type: ogr.Layer

	# create output shape file
	deleteFile(outputPath)
	outputDriver = inputDataSource.GetDriver()
	outputDataSource = outputDriver.CreateDataSource(outputPath)  # type: ogr.DataSource

	outputLayer = outputDataSource.CreateLayer(
		"lines",
		srs=inputLayer.GetSpatialRef(),
		geom_type=ogr.wkbLineString)

	# create attribute fields
	firstFeature = inputLayer.GetNextFeature()  # read first feature to get attributes
	if firstFeature is not None:
		for i in range(firstFeature.GetFieldCount()):
			fieldDef = firstFeature.GetFieldDefnRef(i)
			outputLayer.CreateField(fieldDef)

		inputLayer.ResetReading()
		for feature in inputLayer:  # type: ogr.Feature
			# copy attributes
			outputFeature = ogr.Feature(inputLayer.GetLayerDefn())
			for fieldId in range(0, feature.GetFieldCount()):
				outputFeature.SetField2(fieldId, feature.GetField(fieldId))
			# create geometry
			if feature.geometry() is None:
				continue
			geom = feature.geometry().GetGeometryRef(0)  # type: ogr.Geometry
			if geom is not None and geom.GetPointCount() > 0:
				line = ogr.Geometry(ogr.wkbLinearRing)
				for i in range(0, geom.GetPointCount()):
					x, y = geom.GetPoint_2D(i)
					line.AddPoint_2D(x, y)
				outputFeature.SetGeometry(line)
				outputLayer.CreateFeature(outputFeature)

	# save and cleanup
	outputDataSource.SyncToDisk()
	outputDataSource = None
	inputDataSource = None


def calcOOBB(convexHullPoints):
	"""
	Find the smallest bounding rectangle for a set of points.
	Returns a set of points representing the corners of the bounding box, the rotation angle and matrix.

	based on http://stackoverflow.com/questions/13542855/python-help-to-implement-an-algorithm-to-find-the-minimum-area-rectangle-for-gi/33619018#33619018

	:param convexHullPoints: a Nx2 matrix of convex hull coordinates
	:type convexHullPoints: list | np.array

	:returns bounding box, angle, rotation matrix
	:rtype np.array, float, np.array
	"""
	pi2 = np.pi / 2.

	# calculate edge angles
	edges = np.zeros((len(convexHullPoints) - 1, 2))
	edges = convexHullPoints[1:] - convexHullPoints[:-1]

	angles = np.zeros((len(edges)))
	angles = np.arctan2(edges[:, 1], edges[:, 0])

	angles = np.abs(np.mod(angles, pi2))
	angles = np.unique(angles)

	# find rotation matrices
	rotations = np.vstack([
		np.cos(angles),
		np.cos(angles - pi2),
		np.cos(angles + pi2),
		np.cos(angles)]).T
	rotations = rotations.reshape((-1, 2, 2))

	# apply rotations to the hull
	rotationPoints = np.dot(rotations, convexHullPoints.T)

	# find the bounding points
	minXs = np.nanmin(rotationPoints[:, 0], axis=1)
	maxXs = np.nanmax(rotationPoints[:, 0], axis=1)
	minYs = np.nanmin(rotationPoints[:, 1], axis=1)
	maxYs = np.nanmax(rotationPoints[:, 1], axis=1)

	# find the box with the best area
	areas = (maxXs - minXs) * (maxYs - minYs)
	bestIdx = np.argmin(areas)

	# return the best box
	maxX = maxXs[bestIdx]
	minX = minXs[bestIdx]
	maxY = maxYs[bestIdx]
	minY = minYs[bestIdx]
	r = rotations[bestIdx]
	angle = angles[bestIdx]

	# bb = np.zeros((4, 2))
	# bb[0] = np.dot([minX, minY], r)
	# bb[1] = np.dot([minX, maxY], r)
	# bb[2] = np.dot([maxX, maxY], r)
	# bb[3] = np.dot([maxX, minY], r)
	width = maxX - minX
	height = maxY - minY

	return width, height, angle


def getVectorDataFromWFS(url, layerName, extent, outputPath, outputFormat="GeoJSON", epsg=25832, append=False, attributeFilter=""):
	"""
	Creates vector file from given WFS.
	If too many features (>10000) are requested, then the extent will be split up, so the features per extent decreases.
	This is done recursively. Because features can overlap at extent borders, the file will be searched for duplicates.
	These will be removed.

	:param url: url to WFS
	:type url: str
	:param layerName: name of the layer in the WFS which should be downloaded
	:type layerName: str
	:param extent: (minX, minY, maxX, maxY)
	:type extent: tuple[float, float, float, float]
	:param outputPath: save path
	:type outputPath: str
	:param outputFormat: ogr output format string like "GeoJSON" or "ESRI Shapefile", NOTE: the output format should
						allow appending data!
	:type outputFormat: str
	:param epsg: target epsg (in which extent is)
	:type epsg: int
	:param append: if True, the data will append to the output file
	:type append: bool
	:param attributeFilter: filtering of attributes with SQL WHERE like "Amtlicheflaeche > '300' AND Amtlicheflaeche < '2000'"
	:type attributeFilter: str
	"""
	MAX_FEATURE_COUNT = 9999999  # splitting not working anymore, don't know why, maybe its the server:
								# ERROR 1: Layer tlvermgeo:ALKIS_GESAMT_FKZ not found, and CreateLayer not supported by driver.
								# so data will now be caped at 10000

	if not append:
		deleteFile(outputPath)

	# check how many features we want (we can only get 10000)
	wfsDriver = ogr.GetDriverByName('WFS')
	wfs = wfsDriver.Open("WFS:" + url)  # type: ogr.DataSource
	wfsLayer = wfs.GetLayerByName(layerName)  # type: ogr.Layer
	minX = extent[0]
	minY = extent[1]
	maxX = extent[2]
	maxY = extent[3]
	extentWkt = "POLYGON ((" + str(minX) + " " + str(minY) + "," \
															 "" + str(minX) + " " + str(maxY) + ", " \
																								"" + str(
		maxX) + " " + str(maxY) + ", " \
								  "" + str(maxX) + " " + str(minY) + "," \
																	 "" + str(minX) + " " + str(minY) + "))"
	wfsLayer.SetSpatialFilter(ogr.CreateGeometryFromWkt(extentWkt))
	wfsLayer.SetAttributeFilter(attributeFilter)
	featureCount = wfsLayer.GetFeatureCount()
	if featureCount > MAX_FEATURE_COUNT:  # is maximal number of features exceeded, than split extent in 4 parts
		print "Too many features in WFS request %d, splitting up extent" % featureCount
		filePathWithoutExtension, fileExtension = os.path.splitext(outputPath)
		tempPath = filePathWithoutExtension + "_temp" + fileExtension

		wfs = None
		halfWidth = (maxX - minX) / 2
		halfHeight = (maxY - minY) / 2
		# lower left
		getVectorDataFromWFS(
			url,
			layerName=layerName,
			extent=(minX, minY, minX + halfWidth, minY + halfHeight),
			outputPath=tempPath,
			outputFormat=outputFormat,
			append=True,
			attributeFilter=attributeFilter
		)
		# lower right
		getVectorDataFromWFS(
			url,
			layerName=layerName,
			extent=(minX + halfWidth, minY, maxX, minY + halfHeight),
			outputPath=tempPath,
			outputFormat=outputFormat,
			append=True,
			attributeFilter=attributeFilter
		)
		# upper left
		getVectorDataFromWFS(
			url,
			layerName=layerName,
			extent=(minX, minY + halfHeight, minX + halfWidth, maxY),
			outputPath=tempPath,
			outputFormat=outputFormat,
			append=True,
			attributeFilter=attributeFilter
		)
		# upper right
		getVectorDataFromWFS(
			url,
			layerName=layerName,
			extent=(minX + halfWidth, minY + halfHeight, maxX, maxY),
			outputPath=tempPath,
			outputFormat=outputFormat,
			append=True,
			attributeFilter=attributeFilter
		)
		# remove duplicate entries from overlapping bounds
		tempDataSource = ogr.Open(tempPath)  # type: ogr.DataSource
		tempLayer = tempDataSource.GetLayer()  # type: ogr.Layer
		geometries = []
		for feature in tempLayer:  # type: ogr.Feature
			geometries.append(feature.geometry().ExportToWkt())

		uniqueGeometriesWkt = [attribute for attribute, count in Counter(geometries).items() if count > 1]

		# create final output file and add unique features
		outputDriver = ogr.GetDriverByName(outputFormat)
		outputDataSource = outputDriver.CreateDataSource(outputPath)  # type: ogr.DataSource
		outputLayer = outputDataSource.CreateLayer(
			"cadastre",
			srs=tempLayer.GetSpatialRef(),
			geom_type=tempLayer.GetGeomType()
		)  # type: ogr.Layer
		tempLayer.ResetReading()
		for feature in tempLayer:  # type: ogr.Feature
			geomWkt = feature.geometry().ExportToWkt()
			if geomWkt in uniqueGeometriesWkt:
				uniqueGeometriesWkt.remove(geomWkt)
				outputLayer.CreateFeature(feature.Clone())
		outputDataSource = None
		tempDataSource = None
		deleteFile(tempPath)
		return
	wfs = None
	subprocess.call(
		[
			"ogr2ogr",
			"-f", outputFormat,
			"-t_srs", "EPSG:%d" % epsg,
			"-spat", str(extent[0]), str(extent[1]), str(extent[2]), str(extent[3]),
			outputPath,
					  "WFS:" + url,
			"-append",
			"-where", attributeFilter,
			layerName
		],
		startupinfo=getSubprocessStartUpInfo()
	)


def getPointsGreaterThanThresholdInRasterImage(outputPath, vectorPath, rasterData, rasterGeoTransform, pixelRadius=5, threshold=0):
	"""
	Samples rasterData with given points of a vector file and looks if these samples are higher than a given threshold.

	For each point in the vectorPath file do:
		- get position of point in the rasterData
		- at this position create a window with the pixelRadius
		- calculate the mean of all pixels in this window
		- if this mean is create than the given threshold, the point of the vectorPath file will be copied to output file

	:param outputPath:
	:type outputPath: str
	:param vectorPath: path to point vector data
	:type vectorPath: str
	:param rasterData:
	:type rasterData: np.ndarray
	:param rasterGeoTransform:
	:type rasterGeoTransform: tuple
	:param pixelRadius:
	:type pixelRadius: int
	:param threshold:
	:type threshold:
	"""
	vectorDataSource = ogr.Open(vectorPath, gdal.GA_Update)  # type: ogr.DataSource
	vectorLayer = vectorDataSource.GetLayerByIndex(0)  # type: ogr.Layer

	# create output shape file
	deleteFile(outputPath)
	outputDriver = vectorDataSource.GetDriver()
	outputDataSource = outputDriver.CreateDataSource(outputPath)  # type: ogr.DataSource
	outputLayer = outputDataSource.CreateLayer(
		"points",
		srs=vectorLayer.GetSpatialRef(),
		geom_type=ogr.wkbPoint
	)

	for feature in vectorLayer:  # type: ogr.Feature
		xCoord, yCoord = feature.geometry().GetPoint_2D(0)
		xRasterPos, yRasterPos = coord2pixel(rasterGeoTransform, xCoord, yCoord)
		if 0 <= xRasterPos < rasterData.shape[1] and 0 <= yRasterPos < rasterData.shape[0]:
			imageCutout = rasterData[yRasterPos - pixelRadius:yRasterPos + pixelRadius,
						  xRasterPos - pixelRadius:xRasterPos + pixelRadius]  # type: np.ndarray
			mean = imageCutout.mean()
			if mean >= threshold:
				outputLayer.CreateFeature(feature.Clone())
			# else:
			# 	vectorLayer.DeleteFeature(feature.GetFID())

	# save and cleanup
	outputDataSource.SyncToDisk()
	outputDataSource = None
	vectorDataSource = None


def createBuffers(inputPath, outputPath, outputFormat="GeoJSON", buffer=1):
	"""
	Buffers input data.

	:param inputPath: path to polygon vector layer
	:type inputPath: str
	:param outputPath: path where raster will be saved
	:type outputPath: str
	:param buffer: buffer amount in epsg units
	:type buffer: float
	"""

	inputDataSource = ogr.Open(inputPath)  # type: ogr.DataSource
	inputLayer = inputDataSource.GetLayerByIndex(0)  # type: ogr.Layer
	layerName = inputLayer.GetName()
	inputDataSource = None
	deleteFile(outputPath)
	subprocess.call(
		[
			"ogr2ogr",
			"-dialect", "sqlite",
			"-f", outputFormat,
			"-sql", "SELECT ST_Buffer( geometry , " + str(buffer) + " ),* FROM '" + layerName + "' ",
			outputPath,
			inputPath
		],
		startupinfo=getSubprocessStartUpInfo()
	)


def getIntersectingFeatures(inputPath, inputPath2, outputPath, minIntersectionPercent=1, minAreaOfInput2Features=0):
	"""
	Puts feature from input1 in output, if it has a intersecting feature with input2.

	:param inputPath:
	:type inputPath: str
	:param inputPath2:
	:type inputPath2: str
	:param outputPath:
	:param minAreaOfInput2Features: need for too small houses like garden houses
	:type minAreaOfInput2Features: float | int
	:param minIntersectionPercent:
	:type minIntersectionPercent: float | int
	"""
	minIntersectionPercent /= 100.
	vectorDataSource = ogr.Open(inputPath, gdal.GA_ReadOnly)  # type: ogr.DataSource
	vectorLayer = vectorDataSource.GetLayerByIndex(0)  # type: ogr.Layer

	vectorDataSource2 = ogr.Open(inputPath2, gdal.GA_ReadOnly)  # type: ogr.DataSource
	vectorLayer2 = vectorDataSource2.GetLayerByIndex(0)  # type: ogr.Layer

	# create output shape file
	deleteFile(outputPath)
	outputDriver = vectorDataSource.GetDriver()
	outputDataSource = outputDriver.CreateDataSource(outputPath)  # type: ogr.DataSource
	outputLayer = outputDataSource.CreateLayer(
		"intersecting",
		srs=vectorLayer.GetSpatialRef(),
		geom_type=vectorLayer.GetGeomType()
	)

	for feature1 in vectorLayer:  # type: ogr.Feature
		geom1 = feature1.geometry()  # type: ogr.Geometry
		areaOfIntersectingGeometries = 0
		vectorLayer2.SetSpatialFilter(geom1)
		for feature2 in vectorLayer2:  # type: ogr.Feature
			geom2 = feature2.geometry()  # type: ogr.Geometry
			if geom2.Area() > minAreaOfInput2Features:
				if geom1.Intersects(geom2):
					areaOfIntersectingGeometries += geom1.Intersection(geom2).Area()
		if areaOfIntersectingGeometries / geom1.Area() > minIntersectionPercent:
			outputLayer.CreateFeature(feature1.Clone())

	# save and cleanup
	outputDataSource.SyncToDisk()
	outputDataSource = None
	vectorDataSource = None
	vectorDataSource2 = None

