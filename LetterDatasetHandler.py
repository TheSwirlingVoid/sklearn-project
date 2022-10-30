from sklearn import datasets, svm
from sklearn.utils import Bunch
import numpy as np
import pandas
import cv2 as cv

DATASET_FOLDER = "archive/letters"

class LetterDatasetHandler():

	def __init__(self, datafile) -> None:
		self.datafile = datafile
		self.data = None
# ---------------------------------------------------------------------------- #
	def load_data(self) -> None:
		self.data = Bunch()
		
		csv = pandas.read_csv(self.datafile)
		# add folder prefix to prime image reading
		csv["file"] = DATASET_FOLDER+"/"+csv["file"]
		paths = csv["file"].to_numpy()
		#-----------------------------------------------------#
		self.data["features"] = csv.columns.values
		self.data["target_names"] = csv["letter"].unique()

		self.data["images"] = \
			self.__paths_to_image_array(paths)
		self.data["flattened_images"] = \
			self.__flatten_image_array(self.data["images"])
		#-----------------------------------------------------#
# ---------------------------------------------------------------------------- #
	def __paths_to_image_array(self, paths):

		entries = len(paths)
		x_dim = cv.imread(paths[0]).shape[1]
		y_dim = cv.imread(paths[0]).shape[0]
		# BLANK n-dimensional array for images; this is so we don't get a
		# nested list
		img_array = np.zeros(
			(entries, y_dim, x_dim)
		)
		# populate image ndarray with all images
		for idx in range(len(paths)):
			path = paths[idx]
			image = self.__get_gray_img(path)
			np.append(img_array, image)

		return img_array
# ---------------------------------------------------------------------------- #
	def __flatten_image_array(self, img_array):
		return \
		self.data["images"].reshape(
			(img_array.shape[0], img_array.shape[1]*img_array.shape[2])
		)
# ---------------------------------------------------------------------------- #
	def __get_gray_img(self, path) -> None:
		image = cv.imread(path)
		image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
		return image
# ---------------------------------------------------------------------------- #
	def get_data(self) -> Bunch | None:
		return self.data
# ---------------------------------------------------------------------------- #

def main() -> None:

	# TARGET = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z"]

	dh = LetterDatasetHandler(DATASET_FOLDER+".txt")
	dh.load_data()

	data = dh.get_data()

	print(data["flattened_images"].shape)

	# clf = svm.SVC(gamma=0.001, C=100.)
	# clf.fit(data["flattened_images"], data["targets"])

main()