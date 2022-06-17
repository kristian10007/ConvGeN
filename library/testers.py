"""
This module contains test function for datasets using the logistic regression, the support vector
machine and the k-next-neighbourhood algoritm. Additionally it contains a class for storing the
results of the tests.
"""


import sklearn
# needed in function lr
from sklearn.ensemble import RandomForestClassifier 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.ensemble import GradientBoostingClassifier

_tF1 = "f1 score"
_tTN = "TN"
_tTP = "TP"
_tFN = "FN"
_tFP = "FP"
_tFP = "RF"
_tAps = "average precision score"
_tCks = "cohens kappa score"

class TestResult:
    """
    This class represents the result of one test.

    It stores its *title*, a confusion matrix (*con_mat*), the balanced accuracy score (*bal_acc*)
    and the f1 score (*f1*). If given the average precision score is also stored (*aps*).
    """
    def __init__(self, title, labels=None, prediction=None, aps=None):
        """
        Creates an instance of this class. The stored data will be generated from the given values.

        *title* is a text to identify this result.

        *labels* is a /numpy.array/ containing the labels of the test-data-set.

        *prediction* is a /numpy.array/ containing the done prediction for the test-data-set.

        *aps* is a real number representing the average precision score.
        """
        self.title = title
        self.heading = [_tTN, _tTP, _tFN, _tFP, _tF1, _tCks]
        if aps is not None:
            self.heading.append(_tAps)
        self.data = { n: 0.0 for n in self.heading }

        if labels is not None and prediction is not None:
            self.data[_tF1]     = f1_score(labels, prediction)
            self.data[_tCks]    = cohen_kappa_score(labels, prediction)
            conMat = self._enshureConfusionMatrix(confusion_matrix(labels, prediction))
            [[tn, fp], [fn, tp]] = conMat
            self.data[_tTN] = tn
            self.data[_tTP] = tp
            self.data[_tFN] = fn
            self.data[_tFP] = fp

        if aps is not None:
            self.data[_tAps] = aps

    def __str__(self):
        """
        Generates a text representing this result.
        """
        text = ""

        tn = self.data[_tTN]
        tp = self.data[_tTP]
        fn = self.data[_tFN]
        fp = self.data[_tFP]
        text += f"{self.title} tn, fp: {tn}, {fp}\n"
        text += f"{self.title} fn, tp: {fn}, {tp}\n"

        for k in self.heading:
            if k not in [_tTP, _tTN, _tFP, _tFN]:
                text += f"{self.title} {k}: {self.data[k]:.3f}\n"

        return text

    def csvHeading(self):
        return ";".join(self.heading)

    def toCSV(self):
        return ";".join(map(lambda k: f"{self.data[k]:0.3f}", self.heading))

    @staticmethod
    def _enshureConfusionMatrix(c):
        c0 = [0.0, 0.0]
        c1 = [0.0, 0.0]

        if len(c) > 0:
            if len(c[0]) > 0:
                c0[0] = c[0][0]

            if len(c[0]) > 1:
                c0[1] = c[0][1]

        if len(c) > 1 and len(c[1]) > 1:
            c1[0] = c[1][0]
            c1[1] = c[1][1]

        return [c0, c1]

    def copy(self):
        r = TestResult(self.title)
        r.data = self.data.copy()
        r.heading = self.heading.copy()
        return r


    def addMinMaxAvg(self, mma=None):
        if mma is None:
            return (1, self.copy(), self.copy(), self.copy())

        (n, mi, mx, a) = mma

        for k in a.heading:
            if k in self.heading:
                a.data[k] += self.data[k]

        for k in mi.heading:
            if k in self.heading:
                mi.data[k] = min(mi.data[k], self.data[k])

        for k in mx.heading:
            if k in self.heading:
                mx.data[k] = max(mx.data[k], self.data[k])

        return (n + 1, mi, mx, a)

    @staticmethod
    def finishMinMaxAvg(mma):
        if mma is None:
            return (TestResult("?"), TestResult("?"), TestResult("?"))
        else:
            (n, mi, ma, a) = mma
            for k in a.heading:
                if n > 0:
                    a.data[k] = a.data[k] / n
                else:
                    a.data[k] = 0.0
            return (mi, ma, a)

        


def lr(ttd):
    """
    Runs a test for a dataset with the logistic regression algorithm.
    It returns a /TestResult./

    *ttd* is a /library.dataset.TrainTestData/ instance containing data to test.
    """
    checkType(ttd)
    logreg = LogisticRegression(
        C=1e5,
        solver='lbfgs',
        max_iter=10000,
        multi_class='multinomial',
        class_weight={0: 1, 1: 1.3}
        )
    logreg.fit(ttd.train.data, ttd.train.labels)

    prediction = logreg.predict(ttd.test.data)

    prob_lr = logreg.predict_proba(ttd.test.data)
    aps_lr = average_precision_score(ttd.test.labels, prob_lr[:,1])
    return TestResult("LR", ttd.test.labels, prediction, aps_lr)



def knn(ttd):
    """
    Runs a test for a dataset with the k-next neighbourhood algorithm.
    It returns a /TestResult./

    *ttd* is a /library.dataset.TrainTestData/ instance containing data to test.
    """
    checkType(ttd)
    knnTester = KNeighborsClassifier(n_neighbors=10)
    knnTester.fit(ttd.train.data, ttd.train.labels)
    return runTester(ttd, knnTester, "KNN")


def gb(ttd):
    """
    Runs a test for a dataset with the gradient boosting algorithm.
    It returns a /TestResult./

    *ttd* is a /library.dataset.TrainTestData/ instance containing data to test.
    """
    checkType(ttd)
    tester = GradientBoostingClassifier()
    tester.fit(ttd.train.data, ttd.train.labels)
    return runTester(ttd, tester, "GB")



def rf(ttd):
    """
    Runs a test for a dataset with the random forest algorithm.
    It returns a /TestResult./

    *ttd* is a /library.dataset.TrainTestData/ instance containing data to test.
    """
    checkType(ttd)
    tester = RandomForestClassifier()
    tester.fit(ttd.train.data, ttd.train.labels)
    return runTester(ttd, tester, "RF")



def runTester(ttd, tester, name="GAN"):
    prediction = tester.predict(ttd.test.data)
    return TestResult(name, ttd.test.labels, prediction)

def checkType(t):
    if str(type(t)) == "<class 'numpy.ndarray'>":
        return t.shape[0] > 0 and all(map(checkType, t))
    elif str(type(t)) == "<class 'list'>":
        return len(t) > 0 and all(map(checkType, t))
    elif str(type(t)) in ["<class 'int'>", "<class 'float'>", "<class 'numpy.float64'>"]:
        return True
    elif str(type(t)) == "<class 'library.dataset.DataSet'>":
        return checkType(t.data0) and checkType(t.data1)
    elif str(type(t)) == "<class 'library.dataset.TrainTestData'>":
        return checkType(t.train) and checkType(t.test)
    else:
        raise ValueError("expected int, float, or list, dataset of int, float but got " + str(type(t)))
        return False
    
