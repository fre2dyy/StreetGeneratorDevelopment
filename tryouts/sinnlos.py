# import shapefile
#
#
# sf = shapefile.Reader("files/colour/streets.shp")
# shapes = sf.shapes()
# bbox = shapes[3].bbox
#
# print bbox
# print len(shapes)
# print sf.fields
# print sf.records()
#
# s = sf.shape(0)
# print s.__geo_interface__["type"]
#



# Prepare the environment



import sys
from qgis.core import *
from PyQt4.QtGui import *
app = QApplication([])
QgsApplication.setPrefixPath("/usr", True)
QgsApplication.initQgis()

# Prepare processing framework
sys.path.append('C:/OSGeo4W64/apps/qgis/python/plugins/processing') # Folder where Processing is located
from processing.core.Processing import Processing
Processing.initialize()
from processing.tools import *
# import processing
# clipped_roads = processing.runalg('qgis:clip', osm_road_linetrstrings, your_shapefile, None)
# clipped_roads_layer = QgsVectorLayer(clipped_roads['OUTPUT'], 'Clipped OSM Roads', 'ogr')
# QgsMapLayerRegistry.instance().addMapLayer(clipped_roads_layer, False)