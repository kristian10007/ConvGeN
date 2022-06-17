import math

import tensorflow as tf
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import shuffle
from library.timing import timing


class NNSearch:
    def __init__(self, nebSize=5, timingDict=None):
        self.nebSize = nebSize
        self.neighbourhoods = []
        self.timingDict = timingDict
        self.basePoints = []


    def timerStart(self, name):
        if self.timingDict is not None:
            if name not in self.timingDict:
                self.timingDict[name] = timing(name)

            self.timingDict[name].start()

    def timerStop(self, name):
        if self.timingDict is not None:
            if name in self.timingDict:
                self.timingDict[name].stop()

    def neighbourhoodOfItem(self, i):
        return self.neighbourhoods[i]

    def getNbhPointsOfItem(self, index):
        return self.getPointsFromIndices(self.neighbourhoodOfItem(index))

    def getPointsFromIndices(self, indices):
        nmbi = shuffle(np.array([indices]))
        nmb = self.basePoints[nmbi]
        return tf.convert_to_tensor(nmb[0])

    def neighbourhoodOfItemList(self, items, maxCount=None):
        nbhIndices = set()
        duplicates = []
        for i in items:
            for x in self.neighbourhoodOfItem(i):
                if x in nbhIndices:
                    duplicates.append(x)
                else:
                    nbhIndices.add(x)

        nbhIndices = list(nbhIndices)
        if maxCount is not None:
            if len(nbhIndices) < maxCount:
                nbhIndices.extend(duplicates)
            nbhIndices = nbhIndices[0:maxCount]

        return self.getPointsFromIndices(nbhIndices)


    def fit(self, haystack, needles=None, nebSize=None):
        self.timerStart("NN_fit_chained_init")
        if nebSize == None:
            nebSize = self.nebSize

        if needles is None:
            needles = haystack

        self.basePoints = haystack

        neigh = NearestNeighbors(n_neighbors=nebSize)
        neigh.fit(haystack)
        self.timerStop("NN_fit_chained_init")
        self.timerStart("NN_fit_chained_toList")
        self.neighbourhoods = [
                (neigh.kneighbors([x], nebSize, return_distance=False))[0]
                for (i, x) in enumerate(needles)
                ]
        self.timerStop("NN_fit_chained_toList")
        return self
