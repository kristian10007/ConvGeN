import numpy as np


def normSquared(v):
    s = 0
    for x in v:
        s += x * x
    return s

def distSquared(u, v):
    return normSquared(u - v)

def distToCloud(v, cloud):
    di = None
    for p in cloud:
        d = distSquared(v, p)
        if di is None:
            di = d 
        else:
            di = min(di, d)
    return di

def cloudDist(cloudA, cloudB):
    di = None
    dx = None
    for v in cloudA:
        d = distToCloud(v, cloudB)
        if di is None:
            di = d
            dx = d
        else:
            di = min(di, d)
            dx = max(dx, d)
    return (di, dx)
