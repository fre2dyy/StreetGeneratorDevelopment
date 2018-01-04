import json
from qgis.core import *
from qgis.utils import *

from ui.main_window import MainWindow
from utils.files import getPathToTempFile
from utils.qgis_utils import loadVectorLayer, loadRasterLayer
from utils.osm import OsmBuildings, OsmStreets, OsmWater
from utils.vector import vector2raster, clipping, CopyExtentandEPSG


def main():
	app = QgsApplication([], True)
	app.setPrefixPath("C:/OSGeo4W64/apps/qgis", True)
	app.initQgis()
	window = MainWindow()
	window.show()
	window.raise_()
	bb = (635432, 5616219, 635939, 5616575)	# WestSouthEastNorth
#	bb = (633612, 5619141, 634424, 5619713)	# highway

	# # streets
	streetsPath = ("files/cir/streets.shp")
	osmStreets = OsmStreets(extent=bb, extentEPSG=25832)
	osmStreets.getBufferedStreets(savePath=streetsPath)
	loadVectorLayer(streetsPath, "streets")
	streetMaskPath = ("files/cir/streets_mask.tiff")
	vector2raster(bb, streetsPath, streetMaskPath)
	loadRasterLayer(streetMaskPath, "streetsMask")

	# clipping dop with streetMask
	dop = ("files/colour/streets_dop.tif")
	clippedPathTemp = ("files/colour/streets_clipped_temp.tiff")
	clipping(streetMaskPath, dop, clippedPathTemp)

	# copy extent and EPSG from streetMaskPath
	clippedPath = ("files/colour/streets_clipped.tiff")
	CopyExtentandEPSG(streetMaskPath, clippedPathTemp, clippedPath)

	loadRasterLayer(clippedPath, "clippedSteets")

	#
	# # buildings
	# buildingsPath = ("files/buildings.shp")
	# osmBuildings = OsmBuildings(extent=bb, extentEPSG=25832)
	# osmBuildings.getBuildingPolygons(savePath=buildingsPath, onlyWithHeight=False)
	# loadVectorLayer(buildingsPath, "buildings")
	# buildingMaskPath = ("files/buildings.tiff")
	# vector2raster(bb, buildingsPath, buildingMaskPath)
	# loadRasterLayer(buildingMaskPath, "buildingsMask")

	# water
	# waterPath = "files/water.shp"
	# osmWater = OsmWater(extent=bb, extentEPSG=25832)
	# osmWater.getWaterPolygons(savePath=waterPath)
	# loadVectorLayer(waterPath, "water")
	# waterMaskPath = "files/water.tiff"
	# vector2raster(bb, waterPath, waterMaskPath)
	# loadRasterLayer(waterMaskPath, "waterMask")



	# set CRS to the one of image and zoom to extent
	crs = QgsCoordinateReferenceSystem()
	crs.createFromId(25832)
	window.canvas.setDestinationCrs(crs)

	exitcode = app.exec_()
	QgsApplication.exitQgis()
	sys.exit(exitcode)

if __name__ == "__main__":
	main()
