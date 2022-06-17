from library.interfaces import GanBaseClass
from library.dataset import DataSet

from model.synthesizer.ctabgan_synthesizer import CTABGANSynthesizer 
import pandas as pd

import warnings
warnings.filterwarnings("ignore")



class CtabGan(GanBaseClass):
    """
    This is a toy example of a GAN.
    It repeats the first point of the training-data-set.
    """
    def __init__(self, epochs=10, debug=True):
        self.isTrained = False
        self.epochs = epochs
        self.canPredict = False

    def reset(self, _dataSet):
        """
        Resets the trained GAN to an random state.
        """
        self.isTrained = False
        self.synthesizer = CTABGANSynthesizer(epochs = self.epochs) 

    def train(self, dataSet):
        """
        Trains the GAN.

        It stores the data points in the training data set and mark as trained.

        *dataSet* is a instance of /library.dataset.DataSet/. It contains the training dataset.
        We are only interested in the first *maxListSize* points in class 1.
        """
        if dataSet.data1.shape[0] <= 0:
            raise AttributeError("Train: Expected data class 1 to contain at least one point.")

        self.synthesizer.fit(train_data=pd.DataFrame(dataSet.data1))
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

        return self.synthesizer.sample(numOfSamples)
