import numpy as np
from ConvGenLib.ConvGeN import ConvGeN
from ConvGenLib.dataset import DataSet
import sklearn.datasets

wineDataSet = sklearn.datasets.load_wine()
data = wineDataSet['data']
targets = wineDataSet['target']

n_feat = data.shape[1]


minoritySet = data[np.where(targets == 1)]
majoritySet = data[np.where(targets != 1)]

dataSet = DataSet(data0=majoritySet, data1=minoritySet)

gen = ConvGeN(n_feat=n_feat, neb=5, gen=None, neb_epochs=10, maj_proximal=False)
gen.reset(dataSet)
gen.train(dataSet)

syntheticData = gen.generateData(100)
print(syntheticData)
