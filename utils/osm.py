import json
import os

import time
import overpass
import subprocess
from osgeo import ogr, osr

from files import deleteFile
from base_class import getSubprocessStartUpInfo
from vector import convertToEPSG, convertLinesToPolygons


class Osm:
	"""
	Abstract base class for loading osm data.
	"""
	def __init__(self, extent, extentEPSG=25832):
		"""

		:param extent: extent which should be loaded (must be in extentEPSG coordinates)
		:type extent: tuple
		:param extentEPSG: EPSG ov the given extent
		:type extentEPSG: int
		"""
		self.extent = extent
		self.extentEPSG = extentEPSG

	@staticmethod
	def loadGeoJsonFromOverpass(savePath, overpassQuery):
		"""
		Saves OSM Data as GeoJSON using the overpass API.

		:param savePath: .geojson
		:type savePath: str
		:param overpassQuery: overpass query in Overpass QL (@see http://wiki.openstreetmap.org/wiki/Overpass_API)
		:type savePath: str
		"""
		api = overpass.API(debug=False)
		try:
			response = api.Get(overpassQuery, responseformat="geojson")
		except overpass.OverpassError:
			# if too many requests, wait and send again
			print "Too many overpass requests, trying again in 2 sec."
			time.sleep(2)
			Osm.loadGeoJsonFromOverpass(savePath, overpassQuery)
			return

		# save as geojson
		deleteFile(savePath)
		with open(savePath, 'w') as fp:
			json.dump(response, fp)

	def loadGeoJsonFromOverpassAndConvertToExtentEPSG(self, savePath, overpassQuery):
		"""
		Loads buildings from OpenStreetMap.

		:type savePath: str
		:param overpassQuery: overpass query in Overpass QL (@see http://wiki.openstreetmap.org/wiki/Overpass_API)
		:type savePath: str
		"""
		filePathWithoutExtension, fileExtension = os.path.splitext(savePath)
		tempWrongEPSGPath = filePathWithoutExtension + "_temp_4326.geojson"
		self.loadGeoJsonFromOverpass(tempWrongEPSGPath, overpassQuery)

		convertToEPSG(
			inputPath=tempWrongEPSGPath,
			outputPath=savePath,
			outputFormat="GeoJSON",
			toEPSG=self.extentEPSG
		)

		# cleanup
		deleteFile(tempWrongEPSGPath)

	@staticmethod
	def _transformExtent(extent, inputEPSG=25832, outputEPSG=4326):
		"""
		Returns the given extent (which is in inputEPSG) in the outputEPSG-CRS

		:param extent: (minX, minY, maxX, maxY) in inputEPSG crs
		:type extent: tuple
		:param inputEPSG:
		:type inputEPSG: int
		:param outputEPSG:
		:type outputEPSG: int
		:return: extent in outputEPSG
		:rtype: tuple
		"""

		inSpatialRef = osr.SpatialReference()
		inSpatialRef.ImportFromEPSG(inputEPSG)

		outSpatialRef = osr.SpatialReference()
		outSpatialRef.ImportFromEPSG(outputEPSG)

		coordTransform = osr.CoordinateTransformation(inSpatialRef, outSpatialRef)

		minX, minY, _ = coordTransform.TransformPoint(extent[0], extent[1])
		maxX, maxY, _ = coordTransform.TransformPoint(extent[2], extent[3])
		return minX, minY, maxX, maxY


class OsmWater(Osm):
	def __init__(self, extent, extentEPSG=25832):
		Osm.__init__(self, extent, extentEPSG)

	def getWaterPolygons(self, savePath):
		"""
		Loads buildings form OSM and converts them to polygons (they are lines when using overpass api).

		:type savePath: str
		"""
		filePathWithoutExtension, fileExtension = os.path.splitext(savePath)

		# get waterways
		self.getBufferedWaterways(savePath)

		# get lakes, riverbanks, coast lines etc
		tempWaterLakes = filePathWithoutExtension + "_temp_lakes.geojson"
		self.getLakesPolygons(tempWaterLakes)

		# combine buffered waterways with lakes
		subprocess.call([
			"ogr2ogr",
			"-f", "ESRI Shapefile",
			"-update",
			"-append",
			savePath,
			tempWaterLakes],
			startupinfo=getSubprocessStartUpInfo()
		)

		# cleanup
		deleteFile(tempWaterLakes)

	def getBufferedWaterways(self, savePath):
		"""
		Loads waterways form OSM and converts them to buffers (polygons).
		The size of the buffers will be calculated based on the OSM attributes (width, etc)

		:type savePath: str
		"""
		filePathWithoutExtension, fileExtension = os.path.splitext(savePath)

		tempLinesPath = filePathWithoutExtension + "_temp_lines.geojson"
		self.loadWaterLinesFromOSM(tempLinesPath)

		self.createWaterwayBuffers(
			inputPath=tempLinesPath,
			outputPath=savePath)

		# cleanup
		deleteFile(tempLinesPath)

	def getLakesPolygons(self, savePath):
		filePathWithoutExtension, fileExtension = os.path.splitext(savePath)
		tempWaterPolygonLinesPath = filePathWithoutExtension + "_temp_lines.geojson"
		self.loadWaterPolygonsFromOSM(tempWaterPolygonLinesPath)
		convertLinesToPolygons(
			inputPath=tempWaterPolygonLinesPath,
			outputPath=savePath
		)
		# cleanup
		deleteFile(tempWaterPolygonLinesPath)

	@staticmethod
	def createWaterwayBuffers(inputPath, outputPath):
		"""
		Creates buffers of given waterways (vector layer with lines).

		:param inputPath: path to OSM file (.geojson) with correct epsg
		:type inputPath: str
		:param outputPath: shape file that will be created
		:type outputPath: str
		"""
		inputDataSource = ogr.Open(inputPath)  # type: ogr.DataSource
		inputLayer = inputDataSource.GetLayerByIndex(0)  # type: ogr.Layer

		# create output shape file
		outputDriver = ogr.GetDriverByName("ESRI Shapefile")
		deleteFile(outputPath)
		outputDataSource = outputDriver.CreateDataSource(outputPath)  # type: ogr.DataSource
		outputLayer = outputDataSource.CreateLayer(
			"waterBuffers", srs=inputLayer.GetSpatialRef(),
			geom_type=ogr.wkbPolygon
		)  # type: ogr.Layer

		# create fields of output layer
		outputFieldName = 'width'
		fieldDefn = ogr.FieldDefn(outputFieldName, ogr.OFTReal)
		fieldDefn.SetPrecision(2)
		fieldDefn.SetWidth(6)
		outputLayer.CreateField(fieldDefn)
		featureDefn = outputLayer.GetLayerDefn()  # type: ogr.FeatureDefn
		for feature in inputLayer:  # type: ogr.Feature
			geom = feature.geometry()  # type: ogr.Geometry
			if geom.IsRing():  # if riverbank
				width = ""
				outputGeom = ogr.Geometry(ogr.wkbPolygon)
				ring = ogr.Geometry(ogr.wkbLinearRing)
				for i in range(0, geom.GetPointCount()):
					x, y = geom.GetPoint_2D(i)
					ring.AddPoint_2D(x, y)
				outputGeom.AddGeometry(ring)
			else:
				width = feature.GetFieldAsString("width")
				if width is not None and width != "":
					width = float(width)
				else:
					width = 1
				outputGeom = geom.Buffer(distance=width / 2.0)
			outputFeature = ogr.Feature(featureDefn)
			outputFeature.SetField("width", width)
			outputFeature.SetGeometry(outputGeom)
			outputLayer.CreateFeature(outputFeature)

		# save and cleanup
		outputDataSource.SyncToDisk()
		outputDataSource = None
		inputDataSource = None

	@staticmethod
	def combineBufferedWaterwaysWithLakes(savePath, waterwayBuffersPath, lakesPath):
		"""
		Creates buffers of given waterways (vector layer with lines).

		:param savePath: shape file that will be created
		:type savePath: str
		:param waterwayBuffersPath: shape file that will be created
		:type waterwayBuffersPath: str
		:param lakesPath: shape file that will be created
		:type lakesPath: str
		"""
		waterwayBuffersDataSource = ogr.Open(waterwayBuffersPath)  # type: ogr.DataSource
		waterwayBuffersDataLayer = waterwayBuffersDataSource.GetLayerByIndex(0)  # type: ogr.Layer

		lakesDataSource = ogr.Open(lakesPath)  # type: ogr.DataSource
		lakesDataLayer = lakesDataSource.GetLayerByIndex(0)  # type: ogr.Layer

		# create output shape file
		outputDriver = ogr.GetDriverByName("ESRI Shapefile")
		deleteFile(savePath)
		outputDataSource = outputDriver.CreateDataSource(savePath)  # type: ogr.DataSource
		outputLayer = outputDataSource.CreateLayer(
			"water", srs=waterwayBuffersDataLayer.GetSpatialRef(),
			geom_type=ogr.wkbPolygon
		)  # type: ogr.Layer

		# create fields of output layer
		outputFieldName = 'width'
		fieldDefn = ogr.FieldDefn(outputFieldName, ogr.OFTReal)
		fieldDefn.SetPrecision(2)
		fieldDefn.SetWidth(6)
		outputLayer.CreateField(fieldDefn)
		featureDefn = outputLayer.GetLayerDefn()  # type: ogr.FeatureDefn
		# add buffers
		for feature in waterwayBuffersDataLayer:  # type: ogr.Feature
			width = feature.GetFieldAsString("width")
			outputFeature = ogr.Feature(featureDefn)
			outputFeature.SetField("width", width)
			outputFeature.SetGeometry(feature.geometry())
			outputLayer.CreateFeature(outputFeature)

		# add lake data
		for feature in lakesDataLayer:  # type: ogr.Feature
			outputFeature = ogr.Feature(featureDefn)
			outputFeature.SetField("width", "")
			outputFeature.SetGeometry(feature.geometry())
			outputLayer.CreateFeature(outputFeature)

		# save and cleanup
		outputDataSource.SyncToDisk()
		outputDataSource = None
		waterwayBuffersDataSource = None
		lakesDataSource = None

	def loadWaterLinesFromOSM(self, savePath):
		"""
		Loads waterways (rivers) which are not underground from OpenStreetMap.

		:type savePath: str
		"""
		minX, minY, maxX, maxY = self._transformExtent(self.extent, inputEPSG=self.extentEPSG, outputEPSG=4326)
		bb = str((minY, minX, maxY, maxX))  # Note: y must be first here
		query = "way[waterway]%s->.allWaterWays;" % bb
		query += "way[waterway][tunnel=yes]%s->.undergroundWaterWays;" % bb
		query += "(.allWaterWays; - .undergroundWaterWays;);"

		self.loadGeoJsonFromOverpassAndConvertToExtentEPSG(savePath, query)

	def loadWaterPolygonsFromOSM(self, savePath):
		"""
		Loads water polygons like lakes.

		:type savePath: str
		"""
		minX, minY, maxX, maxY = self._transformExtent(self.extent, inputEPSG=self.extentEPSG, outputEPSG=4326)
		bb = str((minY, minX, maxY, maxX))  # Note: y must be first here
		query = "way[natural=water]%s;" % bb
		self.loadGeoJsonFromOverpassAndConvertToExtentEPSG(savePath, query)


class OsmBuildings(Osm):
	def __init__(self, extent, extentEPSG=25832):
		Osm.__init__(self, extent, extentEPSG)

	def getBuildingPolygons(self, savePath, onlyWithHeight=False):
		"""
		Loads buildings form OSM and converts them to polygons (they are lines when using overpass api).

		:type savePath: str
		:param onlyWithHeight: returns only buildings with height information (height, building levels, ..)
		:type onlyWithHeight: bool
		"""
		filePathWithoutExtension, fileExtension = os.path.splitext(savePath)

		tempLinesPath = filePathWithoutExtension + "_temp_lines.geojson"
		self.loadFromOSM(tempLinesPath, onlyWithHeight=onlyWithHeight)

		convertLinesToPolygons(
			inputPath=tempLinesPath,
			outputPath=savePath
		)

		# cleanup
		deleteFile(tempLinesPath)

	def loadFromOSM(self, savePath, onlyWithHeight=False):
		"""
		Loads buildings from OpenStreetMap.

		:type savePath: str
		:param onlyWithHeight: returns only buildings with height information (height, building levels, ..)
		:type onlyWithHeight: bool
		"""
		minX, minY, maxX, maxY = self._transformExtent(self.extent, inputEPSG=self.extentEPSG, outputEPSG=4326)
		bb = str((minY, minX, maxY, maxX))  # Note: y must be first here
		if onlyWithHeight is True:
			# todo: check if there are more values for height calculation
			query = "way[building]['height']%s->.height;" % bb
			query += "way[building]['building:levels']%s->.height;" % bb
			query += "(.levels;.height;);"
		else:
			query = "way[building]%s" % bb

		self.loadGeoJsonFromOverpassAndConvertToExtentEPSG(savePath, query)


class OsmStreets(Osm):
	def __init__(self, extent, extentEPSG=25832):
		Osm.__init__(self, extent, extentEPSG)

	def getBufferedStreets(self, savePath):
		"""
		Loads streets form OSM and converts them to buffers (polygons).
		The size of the buffers will be calculated based on the OSM attributes (street width, sidewalk etc)

		:type savePath: str
		"""
		filePathWithoutExtension, fileExtension = os.path.splitext(savePath)

		tempLinesPath = filePathWithoutExtension + "_temp_lines.geojson"
		self.loadFromOSM(tempLinesPath)

		self.createStreetBuffers(
			inputPath=tempLinesPath,
			outputPath=savePath)

		# cleanup
		# deleteFile(tempLinesPath)

	@staticmethod
	def createStreetBuffers(inputPath, outputPath):
		"""
		Creates buffers of given streets (vector layer with lines).

		:param inputPath: path to OSM file (.geojson) with correct epsg
		:type inputPath: str
		:param outputPath: shape file that will be created
		:type outputPath: str
		"""
		inputDataSource = ogr.Open(inputPath)  # type: ogr.DataSource
		inputLayer = inputDataSource.GetLayerByIndex(0)  # type: ogr.Layer

		# create output shape file
		outputDriver = ogr.GetDriverByName("ESRI Shapefile")
		deleteFile(outputPath)
		outputDataSource = outputDriver.CreateDataSource(outputPath)  # type: ogr.DataSource
		outputLayer = outputDataSource.CreateLayer(
			"streetBuffers", srs=inputLayer.GetSpatialRef(),
			geom_type=ogr.wkbPolygon
		)  # type: ogr.Layer

		# create fields of output layer

		# outputFieldName = 'turn:lanes'
		# fieldDefn = ogr.FieldDefn(outputFieldName, ogr.OFTString)
		# outputLayer.CreateField(fieldDefn)

		outputFieldName = 'highway'
		fieldDefn = ogr.FieldDefn(outputFieldName, ogr.OFTString)
		outputLayer.CreateField(fieldDefn)

		outputFieldName = 'lanes'
		fieldDefn = ogr.FieldDefn(outputFieldName, ogr.OFTString)
		outputLayer.CreateField(fieldDefn)

		outputFieldName = 'width'
		fieldDefn = ogr.FieldDefn(outputFieldName, ogr.OFTReal)
		fieldDefn.SetPrecision(2)
		fieldDefn.SetWidth(6)
		outputLayer.CreateField(fieldDefn)

		outputFieldName = 'sidewalk'
		fieldDefn = ogr.FieldDefn(outputFieldName, ogr.OFTString)
		outputLayer.CreateField(fieldDefn)

		outputFieldName = 'tunnel'
		fieldDefn = ogr.FieldDefn(outputFieldName, ogr.OFTString)
		outputLayer.CreateField(fieldDefn)

		outputFieldName = 'surface'
		fieldDefn = ogr.FieldDefn(outputFieldName, ogr.OFTString)
		outputLayer.CreateField(fieldDefn)


		featureDefn = outputLayer.GetLayerDefn()  # type: ogr.FeatureDefn
		for feature in inputLayer:  # type: ogr.Feature
			geom = feature.geometry()  # type: ogr.Geometry
			osmParameters = OsmStreetParameters(feature)
			streetWidth = osmParameters.getStreetWidth()
			bufferGeom = geom.Buffer(distance=streetWidth / 2.0)
			outputFeature = ogr.Feature(featureDefn)

			outputFeature.SetField("highway", osmParameters.highway)

			highway_select = outputFeature.GetFieldAsString("highway")
			# selection of "highways"
			if highway_select == "motorway" or \
			   highway_select == "motorway_link" or \
			   highway_select == "trunk" or \
			   highway_select == "trunk_link" or \
			   highway_select == "primary" or \
			   highway_select == "primary_link" or \
			   highway_select == "secondary" or \
			   highway_select == "secondary_link" or \
			   highway_select == "tertiary" or \
			   highway_select == "tertiary_link" or \
			   highway_select == "residential":

				outputFeature.SetField("width", streetWidth)
				outputFeature.SetField("sidewalk", osmParameters.sidewalk)
				outputFeature.SetField("lanes", osmParameters.lanes)
				outputFeature.SetField("tunnel", osmParameters.tunnel)
				outputFeature.SetField("surface", osmParameters.surface)
				outputFeature.SetField("turn:lanes", osmParameters.turnlanes)
			else:
				continue

			outputFeature.SetGeometry(bufferGeom)
			outputLayer.CreateFeature(outputFeature)

		invalid = outputFeature.GetFieldAsString("turn:lanes")
		print invalid

		# save and cleanup
		outputDataSource.SyncToDisk()
		outputDataSource = None
		inputDataSource = None

	def loadFromOSM(self, savePath):
		"""
		Loads streets ("highway") from OpenStreetMap and converts them to the epsg of the extent.

		:type savePath: str
		"""
		minX, minY, maxX, maxY = self._transformExtent(self.extent, inputEPSG=self.extentEPSG, outputEPSG=4326)
		boundingBox = str((minY, minX, maxY, maxX))  # Note: y must be first here
		query = "way[highway]%s;" % boundingBox
		self.loadGeoJsonFromOverpassAndConvertToExtentEPSG(savePath, query)


class OsmStreetParameters:
	def __init__(self, feature):
		"""
		Reads out attributes of given feature (feature will no be saved because reference are only temporally)

		:type feature: ogr.Feature
		"""
		self.highway = feature.GetFieldAsString("highway")
		self.lanes = feature.GetFieldAsString("lanes")
		self.sidewalk = feature.GetFieldAsString("sidewalk")
		self.tunnel = feature.GetFieldAsString("tunnel")
		self.surface = feature.GetFieldAsString("surface")
		self.turnlanes = feature.GetFieldAsString("turn" + ":" + "lanes")
		# print self.turnlanes

	def getStreetWidth(self):
		"""
		:rtype: float
		"""
		return self._getLaneWidth() * self._getLaneCount() + self._getSidewalkCount() * self._getSidewalkWidth()

	def _getLaneWidth(self):
		"""
		Interpreted for germany from:
		http://wiki.openstreetmap.org/wiki/DE:Key:highway
		https://de.wikipedia.org/wiki/Richtlinien_f%C3%BCr_die_Anlage_von_Stra%C3%9Fen_%E2%80%93_Querschnitt
		"""
		if self.highway == "motorway":
			return 3.5
		elif self.highway == "motorway_link":
			return 2.75
		elif self.highway == "trunk":
			return 3.5
		elif self.highway == "trunk_link":
			return 2.75
		elif self.highway == "primary":
			return 3.5
		elif self.highway == "primary_link":
			return 3.5
		elif self.highway == "secondary":
			return 3.0
		elif self.highway == "secondary_link":
			return 3.0
		elif self.highway == "tertiary":
			return 2.75
		elif self.highway == "tertiary_link":
			return 2.75
		elif self.highway == "unclassified":
			return 2.75
		elif self.highway == "residential":
			return 2.75
		elif self.highway == "service":
			return 2.75
		elif self.highway == "living_street":
			return 2.75
		else:
			return 2.0

	def _getLaneCount(self):
		"""
		:rtype: int
		"""
		if self.lanes is None or self.lanes == "":
			if self._getLaneWidth() == 2:
				return 1
			else:
				return 2
		else:
			try:
				return float(self.lanes)
			except ValueError:
				return 2

	def _getSidewalkCount(self):
		"""
		http://wiki.openstreetmap.org/wiki/DE:Key:sidewalk
		:rtype: int
		"""
		if self.sidewalk is None or self.sidewalk == "" or self.sidewalk == "none":
			return 0
		elif self.sidewalk == "both":
			return 2
		elif self.sidewalk == "left" or self.sidewalk == "right":  # todo: make difference between left and right sidewalks
			return 1
		else:
			return 0

	@staticmethod
	def _getSidewalkWidth():
		"""
		Values based on https://de.wikipedia.org/wiki/Gehweg
		:rtype: float
		"""
		return 2.5
