import os.path
import json


def dataCache(fileName, dataGenerator, x=None):
    def flatten(z):
        if str(type(z)) == "<class 'numpy.ndarray'>":
            return [flatten(x) for x in z]
        else:
            return float(z)

    if fileName is not None and os.path.exists(fileName):
        print(f"load data from previous session '{fileName}'")
        with open(fileName) as f:
            return json.load(f)
    else:
        d = dataGenerator(x)

        if fileName is not None:
            print(f"save data for '{fileName}'")
            with open(fileName, 'w') as f:
                json.dump({k: flatten(d[k]) for k in d.keys() }, f)

        return d
                
