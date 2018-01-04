import shapefile


sf = shapefile.Reader("files/colour/streets.shp")
shapes = sf.shapes()
bbox = shapes[3].bbox

print bbox
print len(shapes)
print sf.fields
print sf.records()

s = sf.shape(0)
print s.__geo_interface__["type"]

