import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider
import matplotlib.image
import matplotlib.colorbar

# Note: A reference to instance of ParametrizedPlot must be kept (https://github.com/matplotlib/matplotlib/issues/3105).
# 		So a global var is used here.

globalPlots = None


def init():
	"""
	Must be called once. This must be done, because a reference to each instance of ParametrizedPlot must be kept
	(https://github.com/matplotlib/matplotlib/issues/3105) or the sliders won't work anymore.
	All plots will be added to globalPlots.
	"""
	global globalPlots
	if globalPlots is None:
		globalPlots = []
	return globalPlots


class ParametrizedPlot:
	def __init__(
			self,
			function,
			parameters,
			adjustableParameters,
			plotNonParameters=None,
			overlayImage=None,
			autoScaleColormap=True,
			figTitle="",
			showHist=False):
		"""
		Creates a plot with sliders for the adjustableParameters. Changing the sliders will execute the given function
		with the new values and show the result.
		Before using this class, init() must be called!

		Note: A reference to instance of ParametrizedPlot must be kept (https://github.com/matplotlib/matplotlib/issues/3105).

		:param function: function to be called with the given parameters
		:type function: function
		:param parameters: list of Parameters, not adjustable data
		:type parameters: list[Parameters]
		:param adjustableParameters: list of PlotParameters, which will be adjustable
		:type adjustableParameters: list[AdjustableParameters]
		:param plotNonParameters: creates subplots for the given array data
		:type plotNonParameters: list[ArrayParamters]
		:param overlayImage:
		:type overlayImage: None | np.ndarray | ArrayParameter
		:param autoScaleColormap:
		:type autoScaleColormap: bool
		:param figTitle: title of the figure
		:type figTitle: str
		"""
		assert globalPlots is not None, "init() must be called before using this class."

		self.function = function
		self.parameters = parameters
		self.plotNonParameters = plotNonParameters
		adjustableParameters.reverse()  # because drawing axes is from bottom to top
		self.adjustableParameters = adjustableParameters

		if overlayImage is not None:
			if isinstance(overlayImage, ArrayParameter):
				self.overlayImage = overlayImage
			else:
				self.overlayImage = ArrayParameter("overlayImage", overlayImage)
		else:
			self.overlayImage = None
		self.autoScaleColormap = autoScaleColormap
		self.showHist = showHist
		self.alphaSlider = None  # type: Slider
		self.imagePlot = None  # type: matplotlib.image.AxesImage

		# create figure
		self.fig = plt.figure()
		if figTitle == "":
			figTitle = function.__name__
		self.fig.set_label(figTitle)
		self.fig.canvas.manager.set_window_title(figTitle)
		self.fig.set_size_inches(10, 12, forward=True)

		# create sliders
		for i, para in enumerate(self.adjustableParameters):  # type: int, AdjustableParameter
			if isinstance(para, AdjustableArrayParameter):
				name = para.name + " [NxN]"
			else:
				name = para.name

			para.slider = self._addSlider(
				position=i,
				name=name,
				minValue=para.minValue,
				maxValue=para.maxValue,
				initValue=para.initValue
			)
		# add alpha slider if overlay image is given
		if self.overlayImage is not None:
			self.alphaSlider = self._addSlider(
				position=len(self.adjustableParameters),
				name="alpha",
				minValue=0,
				maxValue=1,
				initValue=0.3
			)

		self._update()

		# get a reference to global scope (because sliders out of scope won't work anymore)
		global globalPlots
		globalPlots.append(self)

	def _addSlider(self, position, name, minValue, maxValue, initValue):
		"""
		Creates a slider at given position in the figure.

		:param position:
		:type position: int
		:param name:
		:type name: str
		:param minValue:
		:type minValue: int | float
		:param maxValue:
		:type maxValue: int | float
		:param initValue:
		:type initValue: int | float

		:return:
		:rtype: Slider
		"""
		valueFormat = '%1.2f'
		if isinstance(initValue, int):
			valueFormat = '%d'
		axis = self.fig.add_axes([0.2, 0.01 + 0.02 * position, 0.50, 0.01], name)
		slider = Slider(
			axis,
			name,
			valmin=minValue,
			valmax=maxValue,
			valinit=initValue,
			valfmt=valueFormat,
			dragging=False
		)
		slider.on_changed(self._update)
		return slider

	def _update(self, val=None):
		"""
		Runs self.function with given parameters and shows result in a plot.
		On first run, the axis for the plots will be created.

		:param val: Not used, but necessary, because Sliders callback expects one parameter.
		"""
		parameterDict = {}
		for para in self.adjustableParameters:  # type: AdjustableParameter
			para.setValue(para.slider.val)
			parameterDict[para.name] = para.value
		for para in self.parameters:  # type: Parameter
			parameterDict[para.name] = para.value

		ret = self.function(**parameterDict)
		# if tuple, use only the first element
		if isinstance(ret, tuple):
			ret = ArrayParameter("return", ret[0])
		else:
			ret = ArrayParameter("return", ret)

		if self.imagePlot is None:  # initial setup
			self._setupPlots(ret)
		else:
			self.imagePlot.set_data(ret.value)
			if self.alphaSlider is not None:
				self.imagePlot.set_alpha(self.alphaSlider.val)
			if self.autoScaleColormap:
				self.imagePlot.autoscale()

			if self.showHist is True:
				hist, binEdges = np.histogram(ret.value, 25)
				self.hist.set_ydata(hist)
				axesYLimit = self.hist.axes.get_ylim()
				if axesYLimit[0] > hist.min() or axesYLimit[1] < hist.max():
					self.hist.axes.set_ylim(min(hist.min(), axesYLimit[0]), max(hist.max(), axesYLimit[1]))
			self.fig.canvas.draw()

	def _setupPlots(self, ret):
		"""
		Creates sliders and subplots.
		:type ret: ArrayParameter
		"""
		# calc sizes of plot elements
		arrayParameters = self._getArrayParametersToPlot()
		slidersHeight = 0.01 + 0.02 * (len(self.adjustableParameters) + 2)
		distanceBetweenPlots = 0.03
		imagePlotHeight = 1 - slidersHeight - (
			distanceBetweenPlots * (len(arrayParameters) + 1 + (1 if self.showHist else 0)))
		imagePlotHeight /= len(arrayParameters) + 1 + (1 if self.showHist else 0)

		# start position of first plot
		imagePlotPosition = 1 - (imagePlotHeight + distanceBetweenPlots)

		# input plot
		for para in arrayParameters:
			self.fig.add_axes([0.1, imagePlotPosition, 0.8, imagePlotHeight], para.name)
			para.imshow()
			plt.title(para.name)
			plt.colorbar()
			imagePlotPosition -= (imagePlotHeight + distanceBetweenPlots)

		# histogram
		if self.showHist:
			self.fig.add_axes([0.1, imagePlotPosition, 0.8, imagePlotHeight], "histogram")
			hist, binEdges = np.histogram(ret.value, 25)
			self.hist = plt.plot(hist)[0]  # type: matplotlib.lines.Line2D
			plt.title("histogram")

		# output plot and add overlay image if present
		self.fig.add_axes([0.1, slidersHeight, 0.8, imagePlotHeight], "output")
		if self.overlayImage is not None:
			self.overlayImage.imshow()
		self.imagePlot = ret.imshow()
		plt.colorbar()
		plt.title("output" + (" (+ overlay image)" if self.overlayImage is not None else ""))
		if self.overlayImage is not None:
			self.imagePlot.set_alpha(self.alphaSlider.val)
			self.imagePlot.set_cmap("gray")

	def _getArrayParametersToPlot(self):
		"""
		Returns all ArrayParameters which need to be plotted.

		:rtype: List[utils.parameter_plots.ArrayParameter]
		"""
		arrayParameters = []
		for para in self.parameters:  # type: Parameter
			if isinstance(para, ArrayParameter):
				if para.plot:
					arrayParameters.append(para)
		if self.plotNonParameters is not None:
			for para in self.plotNonParameters:  # type: ArrayParameter
				if isinstance(para, ArrayParameter):
					arrayParameters.append(para)
		return arrayParameters


class Parameter:
	def __init__(self, name, value):
		"""
		Represents a fix parameter, which can't be changed.

		:param name: name of the parameter
		:type name: str
		:param value: value of the parameter (can be any)
		"""
		self._name = name
		self._value = value

	@property
	def name(self):
		return self._name

	@property
	def value(self):
		return self._value


class ArrayParameter(Parameter):
	def __init__(self, name, value, plot=False, copy=False):
		"""
		Represents a fix parameter, which can't be changed.

		:type name: str
		:type value: np.ndarray
		:param plot: creates a subplot for this array it True
		:type plot: bool
		:param copy: if True, self.value will always return a copy of its data, not the data itself.
		:type copy: bool
		"""
		Parameter.__init__(self, name, value)
		self.plot = plot
		self.copy = copy

	@property
	def value(self):
		if self.copy is True:
			return self._value.copy()
		else:
			return self._value

	def imshow(self):
		"""
		Creates a plot with plt.imshow() and shows the value there. If the value is in wrong format, it will be
		transformed.
		:rtype: matplotlib.image.AxesImage
		"""
		if self._value.ndim == 3:
			if np.where(self._value.shape == np.min(self._value.shape))[
				0] == 0:  # if [channels][y][x] -> [y][x][channels]
				return plt.imshow(self._value.transpose(1, 2, 0))
		return plt.imshow(self._value, interpolation="nearest")


class AdjustableParameter(Parameter):
	def __init__(self, name, initValue, minValue, maxValue):
		"""
		An AdjustableParameter will be represented as a slider in the plot.

		:param name: name of the parameter
		:type name: str
		:param initValue: initial value of the slider
		:type initValue: float | int
		:param minValue: minimum value of the slider
		:type minValue: float | int
		:param maxValue: maximum value of the slider
		:type maxValue: float | int
		"""
		Parameter.__init__(self, name, initValue)
		self._minValue = minValue
		self._maxValue = maxValue
		self.slider = None  # type: Slider
		self._valueType = type(initValue)

	@property
	def minValue(self):
		return self._minValue

	@property
	def maxValue(self):
		return self._maxValue

	@property
	def initValue(self):
		return self._value

	def setValue(self, value):
		self._value = self._valueType(value)


class AdjustableArrayParameter(AdjustableParameter):
	def __init__(self, name, initValue, minValue, maxValue, se=None):
		"""
		The value parameter will be interpreted as the size of a square matrix (n x n)
		An AdjustableArrayParameter will be represented as a slider in the plot.

		:param name: name of the parameter
		:type name: str
		:param initValue: initial value of the slider
		:type initValue: float | int
		:param minValue: minimum value of the slider
		:type minValue: float | int
		:param maxValue: maximum value of the slider
		:type maxValue: float | int
		:param se: open cv structure element i.e. cv2.MORPH_ELLIPSE
		:type se: int
		"""
		AdjustableParameter.__init__(self, name, initValue, minValue, maxValue)
		self._se = se

	@Parameter.value.getter
	def value(self):
		"""
		:rtype: np.ndarray
		"""
		if self._value < 1:
			self._value = 1
		if self._se is not None:
			try:
				import cv2
				return cv2.getStructuringElement(self._se, (self._value, self._value))
			except ImportError as e:
				print "Can't import opencv, using default array with ones: ", e
		return np.ones((self._value, self._value))
