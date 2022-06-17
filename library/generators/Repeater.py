"""
This module contains some example Generative Adversarial Networks for testing.

The classes StupidToyPointGan and StupidToyListGan are not really Networks. This classes are used
for testing the interface. Hope your actually GAN will perform better than this two.

The class SimpleGan is a simple standard Generative Adversarial Network.
"""


import numpy as np

from library.interfaces import GanBaseClass


class Repeater(GanBaseClass):
    """
    This is a toy example of a GAN.
    It repeats the first point of the training-data-set.
    """
    def __init__(self):
        self.canPredict = False
        self.isTrained = False
        self.exampleItems = None
        self.nextIndex = 0

    def reset(self, _dataSet):
        """
        Resets the trained GAN to an random state.
        """
        self.isTrained = False
        self.exampleItems = None

    def train(self, dataSet):
        """
        Trains the GAN.

        It stores the data points in the training data set and mark as trained.

        *dataSet* is a instance of /library.dataset.DataSet/. It contains the training dataset.
        We are only interested in the first *maxListSize* points in class 1.
        """
        if dataSet.data1.shape[0] <= 0:
            raise AttributeError("Train: Expected data class 1 to contain at least one point.")

        self.isTrained = True
        self.exampleItems = dataSet.data1.copy()

    def generateDataPoint(self):
        """
        Returns one synthetic data point by repeating the stored list.
        """
        if not self.isTrained:
            raise ValueError("Try to generate data with untrained Re.")

        if self.nextIndex >= self.exampleItems.shape[0]:
            self.nextIndex = 0

        i = self.nextIndex
        self.nextIndex += 1

        return self.exampleItems[i]


    def generateData(self, numOfSamples=1):
        """
        Generates a list of synthetic data-points.

        *numOfSamples* is a integer > 0. It gives the number of new generated samples.
        """
        numOfSamples = int(numOfSamples)
        if numOfSamples < 1:
            raise AttributeError("Expected numOfSamples to be > 0")

        return np.array([self.generateDataPoint() for _ in range(numOfSamples)])
