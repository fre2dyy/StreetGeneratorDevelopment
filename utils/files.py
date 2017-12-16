from osgeo import gdal
from qgis.core import *
from qgis.utils import *
from PyQt4.QtCore import *


def getPathToTemp():
	"""
	Returns path to temp folder.
	If this is a qgis project, the folder will created in the project folder.
	:return:
	"""
	projectFileInfo = QFileInfo(QgsProject.instance().fileName())
	if projectFileInfo.exists() is False:
		try:
			# temp folder in project root (parent of utils)
			tempPath = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
		except NameError:
			print __name__ + "::getPathToTemp() -> can't  access __file__, using empty path"
			tempPath = ""
	else:
		tempPath = projectFileInfo.absolutePath()

	tempPath = os.path.join(tempPath, "temp")

	if os.path.exists(tempPath) is False:
		os.mkdir(tempPath)
	return tempPath


def getPathToTempFile(filename):
	"""
	Returns full path to a file in temp directory.
	Creates temp directory if not present.

	:param filename: name of file in temp directory
	:type filename: str

	:rtype: str
	"""
	return os.path.join(getPathToTemp(), filename)


def deleteFilesContainingName(name):
	"""
	Deletes all files in temp directory with the given name.
	:param name: file name
	:type name: str
	"""
	pathToTempDir = getPathToTemp()
	if os.path.exists(pathToTempDir):
		filesToDelete = [f for f in os.listdir(pathToTempDir) if name in f]
		for filename in filesToDelete:
			os.remove(os.path.join(pathToTempDir, filename))
	else:
		print __name__ + "::deleteFilesContainingName() -> path to delete does not exists: " + pathToTempDir


def deleteFile(path, _tryCount=0):
	"""
	Delete file at given path. If path is a .shp file, all other files will be removed too (.dbf, .shx, .prj)
	:param path:
	:type path: str
	:param _tryCount: INTERNAL USE! determines how often it was tried to delete the file.
	:type _tryCount: int
	"""
	filePathName, fileExtension = os.path.splitext(path)
	if os.path.isfile(path):
		try:
			os.remove(path)
			if fileExtension.lower() == ".shp":
				if os.path.isfile(filePathName + ".dbf"):
					os.remove(filePathName + ".dbf")
				if os.path.isfile(filePathName + ".shx"):
					os.remove(filePathName + ".shx")
				if os.path.isfile(filePathName + ".prj"):
					os.remove(filePathName + ".prj")
		except WindowsError:
			if _tryCount > 3:
				raise
			time.sleep(2)
			_tryCount += 1
			deleteFile(path, _tryCount)


def createFoldersForPath(path):
	"""
	Checks if the folders to the given path exists, and if not, they will be created.
	:param path:
	"""
	name, ext = os.path.splitext(path)
	if ext != "":
		path = os.path.dirname(path)
	if os.path.exists(path) is False:
		os.makedirs(path)


def saveImage(path, data, driver, projection, transform, dataType=None, returnDataSet=False):
	"""
	Saves a image to disk.

	Note: Throws errors when closing python or qgis, but still working. No idea where this bug come from.
			in tryouts/tryout_bug_gdal_save_crash.py shows the problem.
			All files a closed by using None, but its still not working. Maybe a reference is still there.

	:param path: path where to save file
	:type path: str
	:param data: Bands x NxM or NxM
	:type data: np.ndarray
	:param driver: gdal driver
	:type driver: gdal.Driver
	:param projection:
	:type projection: str
	:param transform: (topLeftX, pixelSizeX, 0, topLeftY, 0, -pixelSizeY), i.e. (635000.0, 0.2, 0.0, 5615800.0, 0.0, -0.2)
	:type transform: tuple
	:type dataType: bits per channel, if None the type of the data will be used
	:type dataType: int | None
	:param returnDataSet: If True, the data set will be return, else the data set will be closed (dataSet = None)
	:type returnDataSet: bool
	"""
	if dataType is None:
		# from https://borealperspectives.wordpress.com/2014/01/16/data-type-mapping-when-using-pythongdal-to-write-numpy-arrays-to-geotiff/
		NP2GDAL_CONVERSION = {
			"uint8": 1,
			"int8": 1,
			"uint16": 2,
			"int16": 3,
			"uint32": 4,
			"int32": 5,
			"float32": 6,
			"float64": 7,
			"complex64": 10,
			"complex128": 11,
		}
		dataType = NP2GDAL_CONVERSION[data.dtype.name]

	createFoldersForPath(path)

	if len(data.shape) == 3:
		bandCount = int(data.shape[0])
		sizeY = int(data.shape[1])
		sizeX = int(data.shape[2])
	else:
		bandCount = 1
		sizeY = int(data.shape[0])
		sizeX = int(data.shape[1])

	dataSet = driver.Create(path, sizeX, sizeY, bandCount, dataType)  # type: gdal.Dataset
	dataSet.SetProjection(projection)
	dataSet.SetGeoTransform(transform)
	if bandCount == 1:
		# dataSet.GetRasterBand(1).SetNoDataValue(-15000)
		dataSet.GetRasterBand(1).WriteArray(data)  # causes error when closing window
	else:
		for i in range(1, bandCount + 1):
			# dataSet.GetRasterBand(i).SetNoDataValue(-15000)
			dataSet.GetRasterBand(i).WriteArray(data[i - 1])  # causes error when closing app window
	if returnDataSet is True:
		return dataSet
	else:
		dataSet = None
