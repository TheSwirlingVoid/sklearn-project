import sys

import cv2 as cv
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
from numpy._typing import NDArray
from numpy.random import rand, randint
import pandas
from sklearn import svm
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import Bunch
from datetime import datetime

DATASET_FOLDER = "archive/letters3"

class LetterDatasetHandler():

    def __init__(self, datafile, paths_column_name, targets_column_name) -> None:
        self.datafile = datafile
        self.data = Bunch()

        self.csv = pandas.read_csv(self.datafile)
        self.column_name = paths_column_name
        self.image_paths = self.csv[self.column_name]
        self.targets = self.csv[targets_column_name]

    def load_data(self) -> None:
        """
        Prepare some internal variables to register this dataset in memory.
        """
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

    def __paths_to_image_array(self, paths):
        """
        Takes an array of image paths and return an image array of their read images.
        """
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

    def __add_class_target(self, target) -> None:
        """
        Add a index-sensitive solution to a piece of training data.
        """
        self.data["class_targets"] = \
            np.append(self.data["class_targets"], target)

    def __flatten_image_array(self, img_array):
        """
        Flatten a given array of images that all have identical dimensions.
        """
        return \
        self.data["images"].reshape(
            (img_array.shape[0], img_array.shape[1]*img_array.shape[2])
        )

    def __get_gray_img(self, path) -> None:
        """
        Get the grayscale of an image.
        """
        image = cv.imread(path)
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        return image

    def get_input_data(self) -> Bunch | None:
        """
        Get the image list in the format of an n-dimensional vector list.
        """
        return self.data["flattened_images"]

    def get_targets(self) -> Bunch | None:
        return self.data["class_targets"]

class OutputDisplay():
    def __init__(self, predictor:"Predictor") -> None:
        self.update_predictor(predictor)

    def update_predictor(self, predictor:"Predictor"):
        """
        Assign a new predictor to the display object.
        """
        if predictor.has_predictions() == False:
            print("Predictor does not have output!")
            self.predictor = Predictor([0], [0], 0) # just a blank predictor
        else: 
            self.predictor = predictor
     
    def display_confusion_matrix(self):
        """ 
        Display a confusion matrix for the `Predictor`;
        a grid which lists true and false positives/negatives.
        """ 
        if (self.predictor.has_predictions()):
            cm = confusion_matrix(
                    self.predictor.get_test_solutions(),
                    self.predictor.get_predictions()
                    )
            ConfusionMatrixDisplay(cm).plot()
            plt.show()
        else: 
            print("Failed to display confusion matrix. \
                Predictor does not have output!")

    def print_prediction_accuracy(self) -> None:
        """
        Prints the prediction accuracy of the predictor.
        """
        print("Prediction accuracy: " + str(self.predictor.calculate_percent_accuracy()*100) + "%.")

 
    def display_test_grid(self):
        """
        Display a grid of the test data images and print out their corresponding solution array.
        """
        test_data = self.predictor.get_test_data()
        test_solutions = self.predictor.get_test_solutions()

        # i'm assuming the images are square here
        test_data = test_data.reshape((test_data.shape[0], 32, 32))

        ncols = 4
        nrows = int(len(test_data)/4)
        
        fig = plt.figure()
        grid = ImageGrid(fig, 111, nrows_ncols=(nrows, ncols), axes_pad=0.1)
        for ax, image in zip(grid, test_data): # ignore the stupid error here
            ax.imshow(image)

        print("Predictions: {:30s}".format(str(self.predictor.get_predictions())))
        print("Corresponding Solutions: {:30s}".format(str(test_solutions)))
        plt.show()


class Predictor():
    def __init__(self, input_data, targets, predict_size) -> None:
        self.predictions = np.empty((0))
        self.train_data, self.test_data, self.train_sol, self.test_sol = \
            train_test_split(
                input_data, targets,
                test_size=predict_size,
                random_state=randint(10000) # some random random seed or something idk
            )

    def train_and_predict(self) -> None:
        """
        Train the `Predictor` and predict some randomly selected test data
        out of what was given.
        """
        # All classifiers were tested with around 1500 data entries
        # Number of test entries: 100
        # Accuracy: 0%
        # clf = svm.SVC(gamma=2, C=1)

        # Accuracy: 27% (65% with letters3)
        clf = svm.SVC(kernel='linear', C=0.025)

        # Accuracy: 5%
        # clf = DecisionTreeClassifier(max_depth=10)

        # Accuracy: 3%
        # clf = AdaBoostClassifier()

        # Accuracy: 3%
        # clf = QuadraticDiscriminantAnalysis()

        # Accuracy: 10%
        # clf = LinearDiscriminantAnalysis()

        # thanks sklearn for being almost absolutely useless
        timeStart = datetime.now()

        print("Training with " + str(len(self.train_data)) + " entries")
        clf.fit(self.train_data, self.train_sol)
        print("Training done!")
        self.predictions = clf.predict(self.test_data)

        timeEnd = datetime.now()
        timeDelta = (timeEnd - timeStart)
        print(timeDelta)

    def calculate_percent_accuracy(self) -> float:
        """
        Calculate the accuracy of how many test data entries the `Predictor` gets right.
        This is output in decimal form.
        """
        if self.has_predictions():

            num_correct = 0
            num_incorrect = 0
            sols = self.get_test_solutions()
            preds = self.get_predictions()
            # iterate through every test solution and check if it matches with the predicted's at the index
            for idx in range(len(preds)):
                if preds[idx] == sols[idx]:
                    num_correct+=1
                else:
                    num_incorrect+=1

            total = num_correct+num_incorrect
            return (num_correct/total)
        else:
            return -1 # if there are no predictions

    def get_predictions(self):
        """
        Gets the array of the `Predictor's` predictions.
        """
        return self.predictions

    def get_test_solutions(self):
        """
        Gets the array of the test data's solutions.
        """
        return self.test_sol

    def get_train_solutions(self) -> NDArray:
        """
        Gets the array of the train data's solutions.
        """
        return self.train_sol

    def get_test_data(self) -> NDArray:
        """
        Gets the array of the train data's solutions.
        """
        return self.test_data

    def has_predictions(self) -> bool:
        """
        Checks if the `Predictor` has predictions.
        """
        return (len(self.predictions) != 0)

def main() -> None:

    dh = LetterDatasetHandler(DATASET_FOLDER+".txt", "file", "letter")
    dh.load_data()

    input_data = dh.get_input_data()
    targets = dh.get_targets()

    predictor = Predictor(input_data, targets, 12)
    predictor.train_and_predict()
    output_display = OutputDisplay(predictor)
    output_display.print_prediction_accuracy()
    output_display.display_test_grid()
    output_display.display_confusion_matrix()

main()
