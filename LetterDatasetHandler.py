import sys
from sklearn import datasets, svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.multiclass import OneVsRestClassifier
from sklearn.utils import Bunch
import matplotlib.pyplot as plt
import numpy as np
import pandas
import cv2 as cv

DATASET_FOLDER = "archive/letters2"

class LetterDatasetHandler():

	def __init__(self, datafile) -> None:
		self.datafile = datafile
		self.data = None
		self.csv = None
# ---------------------------------------------------------------------------- #
	def load_data(self) -> None:
		self.data = Bunch()
		
		self.csv = pandas.read_csv(self.datafile)
		# add folder prefix to prime image reading
		self.csv["file"] = DATASET_FOLDER+"/"+self.csv["file"]
		paths = self.csv["file"].to_numpy()
		#-----------------------------------------------------#
		self.data["features"] = self.csv.columns.values
		self.data["class_targets"] = np.empty((0))
		self.data["possible_results"] = self.csv["letter"].unique()

		self.data["images"] = \
			self.__paths_to_image_array(paths)
		self.data["flattened_images"] = \
			self.__flatten_image_array(self.data["images"])
		#-----------------------------------------------------#
# ---------------------------------------------------------------------------- #
	def __paths_to_image_array(self, paths):

		x_dim = cv.imread(paths[0]).shape[1]
		y_dim = cv.imread(paths[0]).shape[0]
		img_array = np.empty((0, x_dim, y_dim))

		# populate image ndarray with all images
		for idx in range(len(paths)):
			path = paths[idx]
			image = self.__get_gray_img(path)
			# only add the image if it fits the dimension requirement
			if image.shape[1] == x_dim and image.shape[0] == y_dim:
				# brackets around image to add a dimension (2->3)
				img_array = np.append(img_array, [image], axis=0)
				self.__add_class_target(self.csv["letter"][idx])

		return img_array
# ---------------------------------------------------------------------------- #
	def __add_class_target(self, target) -> None:
		self.data["class_targets"] = \
			np.append(self.data["class_targets"], target)
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

	dh = LetterDatasetHandler(DATASET_FOLDER+".txt")
	dh.load_data()

	data = dh.get_data()

	npredict = 10

	# estimator
	clf = svm.SVC()

	train_data, test_data, train_sol, test_sol = \
		train_test_split(
			data["flattened_images"], data["class_targets"],
			test_size=npredict,
			random_state=142 # some random random seed or something idk
		)

	# num-of-images samples, num-of-pixels features
	clf.fit(train_data, train_sol)

	print("Training done!")

	print("Possible results:")
	print(data["possible_results"])

	predictions = clf.predict(test_data)

	cm = confusion_matrix(test_sol, predictions)
	cm_display = ConfusionMatrixDisplay(cm).plot()
	plt.show()
# ---------------------------------------------------------------------------- #
	# while True:
	# 	input_num = input("Enter the relative path to an image to predict: ")
	# 	img = cv.imread(input_num)
	# 	img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
	# 	img = [img.flatten()]
		
	# 	result = str(clf.predict(img))
	# 	print("Predicted: " + result)

main()