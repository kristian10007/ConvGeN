"""
This module contains used interfaces for testing the Generative Adversarial Networks.
"""
import numpy as np


class GanBaseClass:
    """
    Base class for the Generative Adversarial Network.
    It defines the interface used by the Exercise class.
    """

    def __init__(self):
        """
        Initializes the class.
        """
        self.canPredict = False

    def reset(self, dataSet):
        """
        Resets the trained GAN to an random state.
        """
        raise NotImplementedError

    def train(self, dataSet):
        """
        Trains the GAN.
        """
        raise NotImplementedError

    def generateDataPoint(self):
        """
        Generates one synthetic data-point.
        """
        return self.generateData(1)[0]

    def generateData(self, numOfSamples=1):
        """
        Generates a list of synthetic data-points.

        *numOfSamples* is an integer > 0. It gives the number of generated samples.
        """
        raise NotImplementedError

    def predict(self, data, limit=0.5):
        """
        Takes a list (numpy array) of data points.
        Returns a list with real values in [0,1] for the propapility
        that a point is in the minority dataset. With:
          0.0: point is in majority set
          1.0: point is in minority set
        """
        return np.array([max(0, min(1, int(x + 1.0 - limit))) for x in self.predictReal(data)])

    def predictReal(self, data):
        raise NotImplemented

    def retrainDiscriminitor(data, labels):
        pass
