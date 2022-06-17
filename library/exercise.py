"""
Class for testing the performance of Generative Adversarial Networks
in generating synthetic samples for datasets with a minority class.
"""


import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

from library.dataset import DataSet, TrainTestData
from library.testers import lr, knn, gb, rf, TestResult, runTester
import json



class Exercise:
    """
    Exercising a test for a minority class extension class.
    """

    def __init__(self, testFunctions=None, shuffleFunction=None, numOfSlices=5, numOfShuffles=5):
        """
        Creates a instance of this class.

        *testFunctions* is a dictionary /(String : Function)/ of functions for testing
        a generated dataset. The functions have the signature:
        /(TrainTestData, TrainTestData) -> TestResult/

        *shuffleFunction* is either None or a function /numpy.array -> numpy.array/
        that shuffles a given array.

        *numOfSlices* is an integer > 0. The dataset given for the run function
        will be divided in such many slices.

        *numOfShuffles* is an integer > 0. It gives the number of exercised tests.
        The GAN will be trained and tested (numOfShuffles * numOfSlices) times.
        """
        self.numOfSlices = int(numOfSlices)
        self.numOfShuffles = int(numOfShuffles)
        self.shuffleFunction = shuffleFunction
        self.debug = print

        self.testFunctions = testFunctions
        if self.testFunctions is None:
            self.testFunctions = {
                "LR": lr,
                "RF": rf,
                "GB": gb,
                "KNN": knn
                }

        self.results = { name: [] for name in self.testFunctions }

        # Check if the given values are in valid range.
        if self.numOfSlices < 0:
            raise AttributeError(f"Expected numOfSlices to be > 0 but got {self.numOfSlices}")

        if self.numOfShuffles < 0:
            raise AttributeError(f"Expected numOfShuffles to be > 0 but got {self.numOfShuffles}")

    def run(self, gan, dataset, resultsFileName=None):
        """
        Exercise all tests for a given GAN.

        *gan* is a implemention of library.interfaces.GanBaseClass.
        It defines the GAN to test.

        *dataset* is a library.dataset.DataSet that contains the majority class
        (dataset.data0) and the minority class (dataset.data1) of data
        for training and testing.
        """

        # Check if the given values are in valid range.
        if len(dataset.data1) > len(dataset.data0):
            raise AttributeError(
                "Expected class 1 to be the minority class but class 1 is bigger than class 0.")

        # Prepare Folder for Images
        if resultsFileName is not None:
            try:
                os.mkdir(resultsFileName)
            except FileExistsError as e:
                pass

        # Reset results array.
        self.results = { name: [] for name in self.testFunctions }

        if gan.canPredict and "GAN" not in self.testFunctions.keys():
            self.results["GAN"] = []

        # If a shuffle function is given then shuffle the data before the
        # exercise starts.
        if self.shuffleFunction is not None:
            self.debug("-> Shuffling data")
            for _n in range(3):
                dataset.shuffleWith(self.shuffleFunction)

        # Repeat numOfShuffles times
        self.debug("### Start exercise for synthetic point generator")
        for shuffleStep in range(self.numOfShuffles):
            stepTitle = f"Step {shuffleStep + 1}/{self.numOfShuffles}"
            self.debug(f"\n====== {stepTitle} =======")

            # If a shuffle function is given then shuffle the data before the next
            # exercise starts.
            if self.shuffleFunction is not None:
                self.debug("-> Shuffling data")
                dataset.shuffleWith(self.shuffleFunction)


            # Split the (shuffled) data into numOfSlices slices.
            # dataSlices is a list of TrainTestData instances.
            #
            # If numOfSlices=3 then the data will be splited in D1, D2, D3.
            # dataSlices will contain:
            # [(train=D2+D3, test=D1), (train=D1+D3, test=D2), (train=D1+D2, test=D3)]
            self.debug("-> Spliting data to slices")
            dataSlices = TrainTestData.splitDataToSlices(dataset, self.numOfSlices)

            # Do a exercise for every slice.
            for (sliceNr, sliceData) in enumerate(dataSlices):
                sliceTitle = f"Slice {sliceNr + 1}/{self.numOfSlices}"
                self.debug(f"\n------ {stepTitle}: {sliceTitle} -------")
                imageFileName = None
                pickleFileName = None
                if resultsFileName is not None:
                    imageFileName = f"{resultsFileName}/Step{shuffleStep + 1}_Slice{sliceNr + 1}.pdf"
                    pickleFileName = f"{resultsFileName}/Step{shuffleStep + 1}_Slice{sliceNr + 1}.json"
                self._exerciseWithDataSlice(gan, sliceData, imageFileName, pickleFileName)

        self.debug("### Exercise is done.")

        for (n, name) in enumerate(self.results):
            stats = None
            for (m, result) in enumerate(self.results[name]):
                stats = result.addMinMaxAvg(stats)
        
            (mi, mx, avg) = TestResult.finishMinMaxAvg(stats)
            self.debug("")
            self.debug(f"-----[ {avg.title} ]-----")
            self.debug("maximum:")
            self.debug(str(mx))
            self.debug("")
            self.debug("average:")
            self.debug(str(avg))
            self.debug("")
            self.debug("minimum:")
            self.debug(str(mi))

        if resultsFileName is not None:
            return self.saveResultsTo(resultsFileName + ".csv")

        return {}

    def _exerciseWithDataSlice(self, gan, dataSlice, imageFileName=None, pickleFileName=None):
        """
        Runs one test for the given gan and dataSlice.

        *gan* is a implemention of library.interfaces.GanBaseClass.
        It defines the GAN to test.

        *dataSlice* is a library.dataset.TrainTestData instance that contains
        one data slice with training and testing data.
        """

        # Start over with a new GAN instance.
        self.debug("-> Reset the GAN")
        gan.reset(dataSlice.train)

        # Train the gan so it can produce synthetic samples.
        self.debug("-> Train generator for synthetic samples")
        gan.train(dataSlice.train)

        # Count how many syhthetic samples are needed.
        numOfNeededSamples = dataSlice.train.size0 - dataSlice.train.size1

        # Add synthetic samples (generated by the GAN) to the minority class.
        if numOfNeededSamples > 0:
            self.debug(f"-> create {numOfNeededSamples} synthetic samples")
            newSamples = gan.generateData(numOfNeededSamples)

            if pickleFileName is not None:
                with open(pickleFileName, 'w') as f:
                    json.dump({
                        "majority": [[float(z) for z in x] for x in dataSlice.train.data0],
                        "minority": [[float(z) for z in x] for x in dataSlice.train.data1],
                        "synthetic": [[float(z) for z in x] for x in newSamples]
                        }, f)

            # Print out an overview of the new dataset.
            plotCloud(dataSlice.train.data0, dataSlice.train.data1, newSamples, outputFile=imageFileName, doShow=False)

            dataSlice.train = DataSet(
                data0=dataSlice.train.data0,
                data1=np.concatenate((dataSlice.train.data1, newSamples))
                )

        # Test this dataset with every given test-function.
        # The results are printed out and stored to the results dictionary.
        if gan.canPredict and "GAN" not in self.testFunctions.keys():
            self.debug(f"-> retrain GAN for predict")
            trainData = np.concatenate((dataSlice.train.data0, dataSlice.train.data1))
            trainLabels  = np.concatenate((np.zeros(len(dataSlice.train.data0)), np.zeros(len(dataSlice.train.data1)) + 1))
            indices = shuffle(np.array(range(len(trainData))))
            trainData = trainData[indices]
            trainLabels = trainLabels[indices]
            indices = None
            gan.retrainDiscriminitor(trainData, trainLabels)
            trainData = None
            trainLabels = None
            self.debug(f"-> test with GAN.predict")
            testResult = runTester(dataSlice, gan)
            self.debug(str(testResult))
            self.results["GAN"].append(testResult)

        for testerName in self.testFunctions:
            self.debug(f"-> test with '{testerName}'")
            testResult = (self.testFunctions[testerName])(dataSlice)
            self.debug(str(testResult))
            self.results[testerName].append(testResult)


    def saveResultsTo(self, fileName):
        avgResults = {}
        with open(fileName, "w") as f:
            for (n, name) in enumerate(self.results):
                if n > 0:
                    f.write("---\n")
    
                f.write(name + "\n")
                isFirst = True
                stats = None
                for (m, result) in enumerate(self.results[name]):
                    if isFirst:
                        isFirst = False
                        f.write("Nr.;" + result.csvHeading() + "\n")

                    stats = result.addMinMaxAvg(stats)

                    f.write(f"{m + 1};" + result.toCSV() + "\n")
            
                (mi, mx, avg) = TestResult.finishMinMaxAvg(stats)
                f.write(f"max;" + mx.toCSV() + "\n")
                f.write(f"avg;" + avg.toCSV() + "\n")
                f.write(f"min;" + mi.toCSV() + "\n")
                avgResults[name] = avg
        return avgResults


def plotCloud(data0, data1, dataNew=None, outputFile=None, title="", doShow=True):
    """
    Does a PCA analysis of the given data and plot the both important axis.
    """

    if data0.shape[0] > 0:
        if data1.shape[0] > 0:
            data = np.concatenate([data0, data1])
        else:
            data = data0
    else:
        data = data1

    # Normalizes the data.
    if dataNew is None:
        data_t = StandardScaler().fit_transform(data)
    else:
        data_t = StandardScaler().fit_transform(np.concatenate([data, dataNew]))


    # Run the PCA analysis.
    pca = PCA(n_components=2)
    pc = pca.fit_transform(data_t)

    fig, ax = plt.subplots(sharex=True, sharey=True)
    fig.set_dpi(600)
    fig.set_figwidth(10)
    fig.set_figheight(10)
    fig.set_facecolor("white")
    ax.set_title(title)

    def doSubplot(m, n, c):
        pca0 = [x[0] for x in pc[m : m + n]]
        pca1 = [x[1] for x in pc[m : m + n]]
        s = ax.scatter(pca0, pca1, c=c)

    m = 0
    n = len(data0)
    labels = []
    if n > 0:
        labels = ["majority", "minority"]
        doSubplot(m, n, "gray")
    else:
        labels = ["data"]
    
    m += n
    n = len(data1)
    doSubplot(m, n, "red")

    if dataNew is not None:
        m += n
        n = len(dataNew)
        labels.append("synthetic")
        doSubplot(m, n, "blue")

    ax.legend(title="", loc='upper left', labels=labels)
    ax.set_xlabel("PCA0")
    ax.set_ylabel("PCA1")
    if doShow: 
        plt.show()

    if outputFile is not None:
        fig.savefig(outputFile)
