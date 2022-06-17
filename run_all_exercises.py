from library.analysis import testSets, generators, runExercise
import os
import threading

maxWorkers = 1
doMultitask = True

nWorker = 0

for dataset in testSets:
    for name in generators.keys():
        if doMultitask:
            nWorker += 1
            if 0 == os.fork():
                print(f"#{nWorker}: start: {name}({dataset})")
                runExercise(dataset, None, name, generators[name])
                print(f"#{nWorker}: end.")
                exit()
            else:
                if nWorker >= maxWorkers:
                    os.wait()
                    nWorker -= 1
        else:
            runExercise(dataset, None, name, generators[name])

while nWorker > 0:
    os.wait()
    nWorker -= 1

