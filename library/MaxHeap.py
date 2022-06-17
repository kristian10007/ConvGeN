
class MaxHeap:
    def __init__(self, maxSize=None, isGreaterThan=None, smalestValue=(-1,0.0)):
        self.heap = []
        self.size = 0
        self.maxSize = maxSize
        self.isGreaterThan = isGreaterThan if isGreaterThan is not None else (lambda a, b: a > b)
        self.smalestValue = smalestValue
        self.indices = set()
        self.wasChanged = False
        self.insert(smalestValue)

    def copy(self):
        c = MaxHeap(maxSize=self.maxSize, isGreaterThan=self.isGreaterThan, smalestValue=self.smalestValue)
        c.heap = self.heap.copy()
        c.size = self.size
        c.indices = self.indices.copy()
        c.wasChanged = self.wasChanged
        return c

    def insert(self, v):
        if self.maxSize is not None and self.size >= self.maxSize:
            return self.replaceMax(v)

        if v[0] in self.indices:
            return False

        self.indices.add(v[0])
        pos = self.size
        self.size += 1
        self.heap.append(v)
        while pos > 0:
            w = self.heap[pos // 2]
            if not self.isGreaterThan(v, w):
                break
            self.heap[pos] = w
            pos = pos // 2
            self.heap[pos] = v
        self.wasChanged = True
        return True


    def childPos(self, pos):
        c = (pos + 1) * 2
        return (c - 1, c)


    def removeMax(self):
        if self.heap == []:
            self.size = 0
            return False
        
        if self.size <= 1:
            self.size = 0
            self.heap = []
            return True
        
        x = self.heap[0]
        self.indices.remove(x[0])

        self.heap[0] = self.heap[-1]
        self.heap = self.heap[:-1]
        self.size -= 1

        x = self.heap[0]
        pos = 0
        size = self.size

        while pos < size:
            (left, right) = self.childPos(pos)

            if left >= size:
                break

            y = self.heap[left]
            if right >= size:
                if self.isGreaterThan(y, x):
                    self.heap[pos] = y
                    self.heap[left] = x
                break

            z = self.heap[right]
            (best, v) = (left, y) if self.isGreaterThan(y, z) else (right, z)

            if not self.isGreaterThan(v, x):
                break

            self.heap[pos] = v
            self.heap[best] = x
            pos = best

        self.wasChanged = True
        return True


    def replaceMax(self, x):
        if self.heap == []:
            self.heap = [x]
            self.size = 1
            self.indices.add(x[0])
            self.wasChanged = True
            return True
        
        if x[0] in self.indices:
            return False

        if self.isGreaterThan(x, self.heap[0]):
            return False

        self.indices.remove((self.heap[0])[0])
        self.indices.add(x[0])
        self.heap[0] = x
        pos = 0
        size = len(self.heap)

        while pos < size:
            (left, right) = self.childPos(pos)

            if left >= size:
                break

            y = self.heap[left]
            if right >= size:
                if self.isGreaterThan(y, x):
                    self.heap[pos] = y
                    self.heap[left] = x
                break

            z = self.heap[right]
            (best, v) = (left, y) if self.isGreaterThan(y, z) else (right, z)

            if not self.isGreaterThan(v, x):
                break

            self.heap[pos] = v
            self.heap[best] = x
            pos = best

        self.wasChanged = True
        return True

    def getMax(self):
        if self.heap == []:
            return self.smalestValue
        return self.heap[0]


    def setMaxSize(self, maxSize):
        self.maxSize = maxSize
        while self.size > maxSize:
            self.removeMax()

    def toArray(self, mapFn=None):
        return list(self.indices)

    def toOrderedArray(self):
        c = self.copy()
        result = []
        while c.size > 0:
            result.append(c.getMax())
            c.removeMax()
        result.reverse()
        return result

    def length(self):
        return self.size
