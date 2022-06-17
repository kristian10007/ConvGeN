from library.exercise import Exercise
from library.dataset import DataSet, TrainTestData
from library.generators import ProWRAS, SimpleGan, Repeater, ConvGeN, CtGAN, CtabGan

import pickle
import numpy as np
import time
import random
import csv
import gzip
import sys
import os
from imblearn.datasets import fetch_datasets


def loadDataset(datasetName):
    def isSame(xs, ys):
        for (x, y) in zip(xs, ys):
            if x != y:
                return False
        return True
    
    def isIn(ys):
        def f(x):
            for y in ys:
                if isSame(x,y):
                    return True
            return False
        return f

    print(f"Load '{datasetName}'")
    if datasetName.startswith("imblearn_"):
        print("from imblearn")
        ds = fetch_datasets()
        myData = ds[datasetName[9:]]
        ds = None

        features = myData["data"]
        labels = myData["target"]
    elif datasetName.startswith("kaggle_"):
        features = []
        labels = []
        c = csv.reader(gzip.open(f"data_input/{datasetName}.csv.gz", "rt")) 
        for (n, row) in enumerate(c):
            # Skip heading
            if n > 0:
                features.append([float(x) for x in row[:-1]])
                labels.append(int(row[-1]))

        features = np.array(features)
        labels = np.array(labels)

    else:
        print("from pickle file")
        pickle_in = open(f"data_input/{datasetName}.pickle", "rb")
        pickle_dict = pickle.load(pickle_in)

        myData = pickle_dict["folding"]
        k = myData[0]

        labels = np.concatenate((k[1], k[3]), axis=0).astype(float)
        features = np.concatenate((k[0], k[2]), axis=0).astype(float)

    label_1 = list(np.where(labels == 1)[0])
    label_0 = list(np.where(labels != 1)[0])
    features_1 = features[label_1]
    features_0 = features[label_0]
    cut = np.array(list(filter(isIn(features_1), features_0)))
    if len(cut) > 0:
        print(f"non empty cut in {datasetName}! ({len(cut)} points)")
    
    ds = DataSet(data0=features_0, data1=features_1)
    print("Data loaded.")
    return ds


def getRandGen(initValue, incValue=257, multValue=101, modulus=65537):
    value = initValue
    while True:
        value = ((multValue * value) + incValue) % modulus
        yield value
            
def genShuffler():
    randGen = getRandGen(2021)

    def shuffler(data):
        data = list(data)
        size = len(data)
        shuffled = []
        while size > 0:
            p = next(randGen) % size
            size -= 1
            shuffled.append(data[p])
            data = data[0:p] + data[(p + 1):]
        return np.array(shuffled)
    return shuffler


def showTime(t):
    s = int(t)
    m = s // 60
    h = m // 60
    d = h // 24
    s = s % 60
    m = m % 60
    h = h % 24
    if d > 0:
        return f"{d} days {h:02d}:{m:02d}:{s:02d}"
    else:
        return f"{h:02d}:{m:02d}:{s:02d}"


def mkDirIfNotExists(name):
    try:
        os.mkdir(name)
    except FileExistsError as e:
        pass

def runExercise(datasetName, resultList, ganName, ganCreator, skipIfCsvExists=True):
    print(f"* Running {ganName} on {datasetName}")
    oldStdOut = sys.stdout
    oldStdErr = sys.stderr
    resultsFileName = f"data_result/{ganName}"

    # Prepare Folder for result data
    mkDirIfNotExists("data_result")
    mkDirIfNotExists(resultsFileName)

    resultsFileName += f"/{datasetName}"

    try:
        os.stat(f"{resultsFileName}.csv")
        if skipIfCsvExists and resultList is None:
            print("  Resultfile exists => skip calculation.")
            return
    except FileNotFoundError as e:
        pass

    sys.stdout = open(resultsFileName + ".log", "w")
    sys.stderr = sys.stdout


    twStart = time.time()
    tpStart = time.process_time()
    print()
    print()
    print("///////////////////////////////////////////")
    print(f"// Running {ganName} on {datasetName}")
    print("///////////////////////////////////////////")
    print()
    data = loadDataset(f"{datasetName}")
    gan = ganCreator(data)
    random.seed(2021)
    shuffler = genShuffler()

    exercise = Exercise(shuffleFunction=shuffler, numOfShuffles=5, numOfSlices=5)
    avg = exercise.run(gan, data, resultsFileName=resultsFileName)

    tpEnd = time.process_time()
    twEnd = time.time()
    
    if resultList is not None:
        resultList[datasetName] = avg

    sys.stdout = oldStdOut
    sys.stderr = oldStdErr

    print(f"  wall time: {showTime(twEnd - twStart)}s, process time: {showTime(tpEnd - tpStart)}")

    
testSets = [
    "folding_abalone_17_vs_7_8_9_10",
    "folding_abalone9-18",
    "folding_car_good",
    "folding_car-vgood",
    "folding_flare-F",
    "folding_hypothyroid",
    "folding_kddcup-guess_passwd_vs_satan",
    "folding_kr-vs-k-three_vs_eleven",
    "folding_kr-vs-k-zero-one_vs_draw",
    "folding_shuttle-2_vs_5",
    "folding_winequality-red-4",
    "folding_yeast4",
    "folding_yeast5",
    "folding_yeast6",
    #"imblearn_webpage",
    #"imblearn_mammography",
    #"imblearn_protein_homo",
    #"imblearn_ozone_level",
    #"kaggle_creditcard"
    ]


generators = { "Repeater":                lambda _data: Repeater()
             , "ProWRAS":                 lambda _data: ProWRAS()
             , "GAN":                     lambda data: SimpleGan(numOfFeatures=data.data0.shape[1])
             , "CTGAN":                   lambda data: CtGAN(data.data0.shape[1])
             , "CTAB-GAN":                lambda _data: CtabGan()
             , "ConvGeN-majority-5":      lambda data: ConvGeN(data.data0.shape[1], neb=5, gen=5)
             , "ConvGeN-majority-full":   lambda data: ConvGeN(data.data0.shape[1], neb=None)
             , "ConvGeN-proximity-5":     lambda data: ConvGeN(data.data0.shape[1], neb=5, gen=5, maj_proximal=True)
             , "ConvGeN-proximity-full":  lambda data: ConvGeN(data.data0.shape[1], neb=None, maj_proximal=True)
             }
