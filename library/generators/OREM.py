import numpy as np
import random

#import matplotlib.pyplot as plt

from library.interfaces import GanBaseClass
from library.dataset import DataSet

#from keras.layers import Dense, Input, Multiply, Flatten, Conv1D, Reshape
#from keras.models import Model
#from keras import backend as K
#from tqdm import tqdm

#import tensorflow as tf
#from tensorflow.keras.optimizers import Adam
#from tensorflow.keras.layers import Lambda

#from sklearn.utils import shuffle

#from library.NNSearch import NNSearch

import warnings
warnings.filterwarnings("ignore")



class OREM(GanBaseClass):
    """
    This is the ConvGeN class. ConvGeN is a synthetic point generator for imbalanced datasets.
    """
    def __init__(self, q=5, debug=False):
        self.isTrained = False
        self.q = q
        self.debug = debug
        self.canPredict = False
        self.minSet = None
        self.majSet = None
        self.arrayA = None


    def reset(self, dataSet):
        """
        Creates the network.

        *dataSet* is a instance of /library.dataset.DataSet/ or None.
        It contains the training dataset.
        It is used to determine the neighbourhood size if /neb/ in /__init__/ was None.
        """
        self.isTrained = False
        self.minSet = None
        self.majSet = None
        self.arrayA = None



    def train(self, dataSet, discTrainCount=5):
        """
        Trains the Network.

        *dataSet* is a instance of /library.dataset.DataSet/. It contains the training dataset.
        
        *discTrainCount* gives the number of extra training for the discriminator for each epoch. (>= 0)
        """
        if dataSet.data1.shape[0] <= 0:
            raise AttributeError("Train: Expected data class 1 to contain at least one point.")

        self.minSet = dataSet.data1
        self.majSet = dataSet.data0

        minSetSize = len(self.minSet)

        data = np.concatenate( [self.minSet, self.majSet] )
        dataSize = len(data)

        def distance(x, y):
            z = x - y
            return np.sum(z * z)

        def calcDists(x):
            z = np.array([x for _ in range(dataSize)])
            d = z - data
            z = None
            return np.sum(np.multiply(d, d), axis=1)

        def discovCMR(i, q):
            x = data[i]
            dists = list(calcDists(x))
            sortedDots = list(range(len(data)))
            sortedDots.sort(key=(lambda k: dists[k]))

            t = -1
            count = 0
            for k, j in enumerate(sortedDots):
                if j >= minSetSize:
                    count += 1
                    if count >= q:
                        t = max(0, k - q)
                        break
                else:
                    count = 0
            return [(i, data[sortedDots[i]], dists[sortedDots[i]]) for i in range(max(1, t + 1))]

        def isClean(xc, r, points):
            for _, z, _ in points:
                if z not in self.minSet and distance(xc, z) <= r:
                    return False
            return True

        def ideCleanReg(i, q):
            a = []
            x = data[i]
            c = discovCMR(i, q)
            for p, y, d in c:
                xc = 0.5 * (x + y)
                rp = 0.5 * d
                if isClean(xc, rp, c[0:p]):
                    a.append(y)
            return(a)

        self.arrayA = [ideCleanReg(i, self.q) for i in range(minSetSize) ]
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
            raise ValueError("Try to generate data with untrained network.")

        synth_set=[]
        n = 0
        while n < numOfSamples:
            for x, a in zip(self.minSet, self.arrayA):
                k = random.randint(0, max(0, len(a) - 1))
                y = a[k]
                gamma = random.random()
                if y in self.majSet:
                    gamma *= 0.5
                synth_set.append(x + (gamma * (y - x)))
                n += 1
                if n >= numOfSamples:
                    break

        return np.array(synth_set)

    def predictReal(self, data):
        """
        Uses the discriminator on data.
        
        *data* is a numpy array of shape (n, n_feat) where n is the number of datapoints and n_feat the number of features.
        """
        return np.array([0  for x in data])

