"""
This module contains classes to collect data for testing and training.
"""


import math
import numpy as np


class DataSet:
    """
    This class stores data and labels for a test or training dataset.

    *data0*, *data1* are instances of /numpy.array/. Containg the data for the class 0 (majority
    class) and the class 1 (minority class).

    *size0*, *size1* are integers, giving the size of the classes 0 and 1.

    *data* is an instance of /numpy.array/ containing the combined classes 0 and 1.

    *labels* is a /numpy.array/ containing the labels for *data*.
    """
    def __init__(self, data0=None, data1=None):
        """
        Initializes one instance of this class and fills *data* and *labels*.
        """
        self.data0 = data0
        self.data1 = data1
        self.size0 = len(data0) if data0 is not None else 0
        self.size1 = len(data1) if data1 is not None else 0

        if data0 is not None and data1 is not None:
            self.data = np.concatenate( [data1, data0] )
            self.labels = np.concatenate( [self.labels1(), self.labels0()] )
        elif data0 is None:
            self.data = data1
            self.labels = self.labels1()
        elif data1 is None:
            self.data = data0
            self.labels = self.labels0()
        else:
            raise AttributeError("Expected data, data0 or data1 to be a numpy.array")

    def shuffleWith(self, shuffleFn):
        """
        Shuffles the points in the classes 0 and 1 with the given function
        (numpy.array -> numpy.array). After that the *data* array will be regenerated.
        """
        if self.data0 is not None:
            self.data0 = shuffleFn(self.data0)

        if self.data1 is not None:
            self.data1 = shuffleFn(self.data1)

        if self.data0 is None:
            self.data = self.data1
        elif self.data1 is None:
            self.data = self.data0
        else:
            self.data = np.concatenate((self.data1, self.data0))

    def labels0(self):
        """
        Returns a /numpy.array/ with labels for class0.
        """
        return np.zeros(self.size0)

    def labels1(self):
        """
        Returns a /numpy.array/ with labels for class1.
        """
        return np.zeros(self.size1) + 1


class TrainTestData:
    """
    Stores data and labels for class 0 and class 1.

    *train* is a /DataSet/ containing the data for training.

    *test* is a /DataSet/ containing the data for testing.
    """

    def __init__(self, train, test):
        """
        Initializes a new instance for this class and stores the given data.
        """
        self.train = train
        self.test = test

    @classmethod
    def splitDataByFactor(cls, features0, features1, factor=0.9):
        """
        Creates a new instance of this class.

        The first (factor * 100%) percent of the points in the given classes are stored for
        training. The remaining points are stored for testing.

        *features0* and *features1* are /numpy.array/ instances containing the data for class 0
        and class 1.

        *factor* is a real number > 0 and < 1 for the spliting point.
        """

        if factor <= 0.0 or factor >= 1.0:
            raise AttributeError(f"Expected trainFactor to be between 0 and 1 but got {factor}.")

        # ----------------------------------------------------------------------------------------
        # Supporting function:
        def splitUpData(data):
            """
            Splits a given /numpy.array/ in two /numpy.array/.
            The first array contains (factor * 100%) percent of the data points.
            The second array contains the remaining data points.
            """
            size = len(data)
            trainSize = math.ceil(size * factor)
            trn = data[list(range(0, trainSize))]
            tst = data[list(range(trainSize, size))]
            return trn, tst
        # ----------------------------------------------------------------------------------------

        features_0_trn, features_0_tst = splitUpData(features0)
        features_1_trn, features_1_tst = splitUpData(features1)

        return cls(
            test=DataSet(data1=features_1_tst, data0=features_0_tst),
            train=DataSet(data1=features_1_trn, data0=features_0_trn)
            )

    @classmethod
    def splitDataToSlices(cls, bigData, numOfSlices=5):
        """
        Creates a list of new instance of this class. The list is returned as a generator.

        The given data is splitted in the given number of slices.

        *bigData* is an instance of /DataSet/ containing the data to split.

        *numOfSlices* is the number of generated slices.
        """

        numOfSlices = int(numOfSlices)
        if numOfSlices < 1:
            raise AttributeError(f"Expected numOfSlices to be positive but got {numOfSlices}")

        # ----------------------------------------------------------------------------------------
        # Supporting function:
        def arrayToSlices(data):
            """
            Takes a /numpy.array/ and splits it into *numOfSlices* slices.
            A list of the slices will be returned.
            """
            size = len(data)
            if size < numOfSlices:
                raise AttributeError(
                    f"Expected data set to contain at least {numOfSlices} points"
                    + f" but got {size} points."
                    )

            sliceSize = (size // numOfSlices) + (0 if size % numOfSlices == 0 else 1)

            return [
                data[n * sliceSize : min(size, (n+1) * sliceSize)]
                for n in range(numOfSlices)
                ]
        # ----------------------------------------------------------------------------------------

        data0slices = arrayToSlices(bigData.data0)
        data1slices = arrayToSlices(bigData.data1)

        for n in range(numOfSlices):
            data0 = np.concatenate([data0slices[k] for k in range(numOfSlices) if n != k])
            data1 = np.concatenate([data1slices[k] for k in range(numOfSlices) if n != k])
            train = DataSet(data0=data0, data1=data1)
            test = DataSet(data0=data0slices[n], data1=data1slices[n])
            yield cls(train=train, test=test)
