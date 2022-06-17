import numpy as np
import ctgan
import math

from library.interfaces import GanBaseClass
from library.dataset import DataSet


class CtGAN(GanBaseClass):
    """
    This is a toy example of a GAN.
    It repeats the first point of the training-data-set.
    """
    def __init__(self, epochs=10, debug=False):
        self.isTrained = False
        self.epochs = epochs
        self.debug = debug
        self.ctgan = None
        self.canPredict = False

    def reset(self, _dataSet):
        """
        Resets the trained GAN to an random state.
        """
        self.isTrained = False
        ## instanciate generator network and visualize architecture
        self.ctgan = ctgan.CTGANSynthesizer(epochs=self.epochs) 

    def train(self, dataSet):
        """
        Trains the GAN.

        It stores the data points in the training data set and mark as trained.

        *dataSet* is a instance of /library.dataset.DataSet/. It contains the training dataset.
        We are only interested in the first *maxListSize* points in class 1.
        """
        if dataSet.data1.shape[0] <= 0:
            raise AttributeError("Train: Expected data class 1 to contain at least one point.")

        discreteColumns = self.findDiscreteColumns(dataSet.data1)

        if discreteColumns != []:
            self.ctgan.fit(dataSet.data1, discreteColumns)
        else:
            self.ctgan.fit(dataSet.data1)
        self.isTrained = True

    def generateDataPoint(self):
        """
        Returns one synthetic data point by repeating the stored list.
        """
        return (self.generateData(1))[0]


    def generateData(self, numOfSamples=1):
        """
        Generates a list of synthetic data-points.

        *numOfSamples* is a integer > 0. It gives the number of new generated samples.
        """
        if not self.isTrained:
            raise ValueError("Try to generate data with untrained Re.")

        return self.ctgan.sample(numOfSamples)


    def findDiscreteColumns(self, data):
        columns = set(range(data.shape[1]))

        for row in data:
            for c in list(columns):
                x = row[c]
                if float(math.floor(x)) != x:
                    columns.remove(c)

            if len(columns) == 0:
                break

        return columns
