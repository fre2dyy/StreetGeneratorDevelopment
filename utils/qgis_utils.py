from qgis.core import *
from qgis.utils import *
from PyQt4.QtCore import *
from files import getPathToTempFile, deleteFilesContainingName, deleteFile


def getFirstFeatureOfLayer(layer):
	"""
	:type layer: QgsVectorLayer
	"""
	for feat in layer.getFeatures():
		return feat


def removeMapLayersByName(name):
	"""
	Removes all layers with the given name.
	:type name: str
	"""
	layers = QgsMapLayerRegistry.instance().mapLayersByName(name)  # type: QgsMapLayer
	QgsMapLayerRegistry.instance().removeMapLayers([layer.id() for layer in layers])


def removeMapLayersByPath(path):
	"""
	Removes layer with the given path.
	:type path: str
	"""
	layers = QgsMapLayerRegistry.instance().mapLayers()  # type: dict
	for layerName, layer in layers.items():  # type: QgsMapLayer
		if layer.source() == path:
			QgsMapLayerRegistry.instance().removeMapLayer(layer)


def createAndAddLayer(geomType, name, memType="memory", deleteOldLayerFile=False, transparency=0, crs=25832):
	"""
	Creates a QGIS layer and adds the layer to QgsMapLayerRegistry.

	:param geomType: geometry type, valid values: 'polygon', 'linestring', 'point'
	:type geomType: str
	:param name: name of the layer
	:type name: str
	:param memType: 'memory' for layer in memory or 'ogr' for shape file on disk in temp folder
	:type memType: str
	:param deleteOldLayerFile: if True, temporary data will be deleted first
	:type deleteOldLayerFile: bool
	:param transparency: layer transparency
	:type transparency: float
	:param crs: epsg id of coordinate reference system
	:type crs: int

	:rtype: QgsVectorLayer
	"""
	layer = createLayer(geomType, name, memType, deleteOldLayerFile, crs)
	addLayerToRegistry(layer, transparency)
	return layer


def createLayer(geomType, name, memType="memory", deleteOldLayerFile=False, crs=25832, path=None,
				driverName="ESRI Shapefile"):
	"""
	Creates a QGIS layer and adds the layer.

	:param geomType: geometry type, valid values: 'polygon', 'linestring', 'point'
	:type geomType: str
	:param name: name of the layer
	:type name: str
	:param memType: 'memory' for layer in memory or 'ogr' for shape file on disk in temp folder
	:type memType: str
	:param deleteOldLayerFile: if True, temporary data will be deleted first
	:type deleteOldLayerFile: bool
	:param crs: epsg id of coordinate reference system
	:type crs: int

	:rtype: QgsVectorLayer
	"""
	geomType = geomType.lower()
	memType = memType.lower()
	removeMapLayersByName(name)
	if memType == "memory":
		layer = QgsVectorLayer(geomType + "?crs=epsg:" + str(crs), name, memType)
	elif memType == "ogr":
		if path is None:
			if driverName == "GeoJSON":
				path = getPathToTempFile(name + ".geojson")
			else:
				path = getPathToTempFile(name + ".shp")
		if QFileInfo(path).exists() is False or deleteOldLayerFile is True:
			if deleteOldLayerFile is True:
				deleteFile(path)
			# deleteFilesContainingName(name)

			if geomType == "polygon":
				geomType = QGis.WKBPolygon
			elif geomType == "linestring":
				geomType = QGis.WKBLineString
			elif geomType == "point":
				geomType = QGis.WKBPoint
			else:
				raise NotImplementedError("Only polygon, linestring and point support for now, not " + memType)

			fields = QgsFields()
			fields.append(QgsField("id", QVariant.Int))
			writer = QgsVectorFileWriter(path, "CP1250", fields, geomType, QgsCoordinateReferenceSystem(crs),
										 driverName)
			print writer.errorMessage()
			del writer
		else:
			print "Using existing file: " + path
		layer = QgsVectorLayer(path, name, "ogr")
	else:
		raise TypeError("MemoryType not found")
	return layer


def addLayerToRegistry(layer, transparency=0):
	"""
	Adds the layer to QgsMapLayerRegistry.

	:param layer: layer transparency
	:type layer: QgsMapLayer
	:param transparency: layer transparency
	:type transparency: float

	:rtype: QgsMapLayer
	"""
	QgsMapLayerRegistry.instance().addMapLayer(layer, False)
	# reference to the layer tree
	root = QgsProject.instance().layerTreeRoot()
	# adds the memory layer to the layer node at index 0
	layerNode = QgsLayerTreeLayer(layer)
	root.insertChildNode(0, layerNode)
	# set custom property
	if isinstance(layer, QgsVectorLayer):
		layerNode.setCustomProperty("showFeatureCount", True)
	setTransparencyOfLayer(layer, transparency)
	return layer


def loadVectorLayer(path, name=None, memType="ogr", transparency=0):
	"""
	Loads a QGIS layer from file and adds the layer to QgsMapLayerRegistry.

	:param path: path to file
	:type path: str
	:param name: name of the layer
	:type name: str
	:param memType: 'ogr' for shape file on disk in temp folder
	:type memType: str
	:param transparency: layer transparency
	:type transparency: float


	:rtype: QgsVectorLayer
	"""
	if name is None:
		filePathWithoutExtension, fileExtension = os.path.splitext(path)
		name = os.path.basename(filePathWithoutExtension)
	memType = memType.lower()
	removeMapLayersByPath(path)
	layer = QgsVectorLayer(path, name, memType)

	addLayerToRegistry(layer, transparency)
	return layer


def loadRasterLayer(path, name=None, rasterType="gdal", transparency=0, removeLayersWithSameName=False):
	"""
	Creates a QGIS layer and adds the layer to QgsMapLayerRegistry.

	:type path: str
	:param name: name of the layer, if None path name will be used
	:type name: str | None
	:type rasterType: str
	:param transparency: layer transparency
	:type transparency: float
	:param removeLayersWithSameName: if True, all layers with same name as the given name will be deleted
	:type removeLayersWithSameName: bool

	:rtype: QgsRasterLayer
	"""
	if name is None:
		filePathWithoutExtension, fileExtension = os.path.splitext(path)
		name = os.path.basename(filePathWithoutExtension)

	rasterType = rasterType.lower()
	if removeLayersWithSameName:
		removeMapLayersByName(name)
	layer = QgsRasterLayer(path, name, rasterType)

	addLayerToRegistry(layer, transparency)
	return layer


def createAndAddFeature(layer, geom):
	"""
	Creates a feature with given geometry and adds it to layer.

	:type layer: QgsVectorLayer
	:type geom: QgsGeometry
	:rtype: QgsFeature
	"""
	feat = QgsFeature()
	feat.setGeometry(geom)
	layer.dataProvider().addFeatures([feat])
	layer.triggerRepaint()
	return feat


def findFeaturesInLayerByBoundingBox(boundingBox, layer):
	"""
	:type boundingBox: QgsRectangle
	:type layer: QgsVectorLayer
	:rtype: list[QgsFeature]
	"""
	req = QgsFeatureRequest(boundingBox)
	reqIter = layer.getFeatures(req)
	retList = []
	for feat in reqIter:
		retList.append(feat)
	return retList


def setTransparencyOfLayer(layer, transparency):
	"""
	:param layer:
	:type layer: QgsMapLayer
	:param transparency:
	:type transparency: float
	"""
	if isinstance(layer, QgsVectorLayer):
		if transparency > 0:
			colorEffect = QgsColorEffect()
			colorEffect.setTransparency(transparency)
			layer.rendererV2().setPaintEffect(colorEffect)
	elif isinstance(layer, QgsRasterLayer):
		opacity = 1 - transparency
		if opacity < 1:
			layer.renderer().setOpacity(opacity)


def findIntersectingFeatures(layer1, layer2, minIntersectionPercent=0):
	"""
	Creates new layer and adds features of layer1 that intersects with features of layer 2.

	:type layer1: QgsVectorLayer
	:type layer2: QgsVectorLayer
	:type minIntersectionPercent: float | int
	"""
	minIntersectionPercent /= 100.
	layer = createAndAddLayer("Polygon", "Intersecting", crs=25832, transparency=0.3)  # type: QgsVectorLayer
	for feature1 in layer1.getFeatures():  # type: QgsFeature
		geom1 = feature1.geometry()  # type: QgsGeometry
		areaOfIntersectingGeometries = 0
		for feature2 in layer2.getFeatures(QgsFeatureRequest(geom1.boundingBox())):  # type: QgsFeature
			geom2 = feature2.geometry()  # type: QgsGeometry
			if geom1.intersects(geom2):
				areaOfIntersectingGeometries += geom1.intersection(geom2).area()
		if areaOfIntersectingGeometries / geom1.area() > minIntersectionPercent:
			createAndAddFeature(layer, geom1)


def removeIntersectingFeatures(pointLayer, polygonLayer):
	"""
	Removes all points of points layer that interact with polygon layer.

	:type pointLayer: QgsVectorLayer
	:type polygonLayer: QgsVectorLayer
	"""
	featureIdsToRemove = []
	for polygonFeature in polygonLayer.getFeatures():  # type: QgsFeature
		polygonGeometry = polygonFeature.geometry()  # type: QgsGeometry
		for pointFeature in pointLayer.getFeatures(
				QgsFeatureRequest(polygonGeometry.boundingBox())):  # type: QgsFeature
			pointGeometry = pointFeature.geometry()  # type: QgsGeometry
			if pointGeometry.intersects(polygonGeometry):
				featureIdsToRemove.append(pointFeature.id())
	print "removing %d features" % len(featureIdsToRemove)
	pointLayer.deleteFeatures(featureIdsToRemove)


def createPointsAlongLines(inputPath, outputPath, distanceBetweenPoints=1):
	"""
	Creates Points along a given line.
	Based on http://gis.stackexchange.com/questions/27102/creating-equidistant-points-in-qgis

	:type inputPath: str
	:type outputPath: str
	:param distanceBetweenPoints: distance between points in given epsg units
	:type distanceBetweenPoints: float
	"""
	inputLayer = loadVectorLayer(inputPath)  # type: QgsVectorLayer

	# create output shape file
	outputLayer = createLayer("point", name="points along line", memType="ogr", deleteOldLayerFile=True,
							  path=outputPath)  # type: QgsVectorLayer

	for feature in inputLayer.getFeatures():  # type: QgsFeature
		# create geometry
		geom = feature.geometry()  # type: QgsGeometry
		length = geom.length()
		currentDistance = 0
		featurePoints = []
		while currentDistance <= length:
			point = geom.interpolate(currentDistance)
			outputFeature = QgsFeature()
			outputFeature.setGeometry(point)
			featurePoints.append(outputFeature)
			currentDistance += distanceBetweenPoints
		outputLayer.dataProvider().addFeatures(featurePoints)
	outputLayer.triggerRepaint()
	removeMapLayersByPath(inputPath)
