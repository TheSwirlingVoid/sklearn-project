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

DATASET_FOLDER = "archive/letters3"

class LetterDatasetHandler():

	def __init__(self, datafile, paths_column_name, targets_column_name) -> None:
		self.datafile = datafile
		self.data = Bunch()

		self.csv = pandas.read_csv(self.datafile)
		self.column_name = paths_column_name
		self.image_paths = self.csv[self.column_name]
		self.targets = self.csv[targets_column_name]
# ---------------------------------------------------------------------------- #
	def load_data(self) -> None:
		# add folder prefix to prime image reading
		self.image_paths = DATASET_FOLDER+"/"+self.image_paths
		paths = self.image_paths.to_numpy()
		#-----------------------------------------------------#
		self.data["features"] = self.csv.columns.values
		self.data["class_targets"] = np.empty((0))

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
				self.__add_class_target(self.targets[idx])

		return img_array
# ---------------------------------------------------------------------------- #
	def __add_class_target(self, target) -> None:
		self.data["class_targets"] = \
			np.append(self.data["class_targets"], target)
# ---------------------------------------------------------------------------- #
	def __flatten_image_array(self, img_array):
		return \
		self.data["images"].reshape(
			(img_array.shape[0], None)
		)
# ---------------------------------------------------------------------------- #
	def __get_gray_img(self, path) -> None:
		image = cv.imread(path)
		image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
		return image
# ---------------------------------------------------------------------------- #
	def get_input_data(self) -> Bunch | None:
		return self.data["flattened_images"]
# ---------------------------------------------------------------------------- #
	def get_targets(self) -> Bunch | None:
		return self.data["class_targets"]
# ---------------------------------------------------------------------------- #

def main() -> None:

	dh = LetterDatasetHandler(DATASET_FOLDER+".txt", "file", "letter")
	dh.load_data()

	input_data = dh.get_input_data()
	targets = dh.get_targets()

	npredict = 100

	# estimator
	clf = svm.SVC()

	train_data, test_data, train_sol, test_sol = \
		train_test_split(
			input_data, targets,
			test_size=npredict,
			random_state=142 # some random random seed or something idk
		)

	print("Training with " + str(len(train_data)) + " entries")
	# num-of-images samples, num-of-pixels features
	clf.fit(train_data, train_sol)

	print("Training done!")

	predictions = clf.predict(test_data)

	cm = confusion_matrix(test_sol, predictions)
	ConfusionMatrixDisplay(cm).plot()
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