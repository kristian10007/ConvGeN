from library.ext_prowras import ProWRAS_gen
from library.interfaces import GanBaseClass


class ProWRAS(GanBaseClass):
    """
    This is a toy example of a GAN.
    It repeats the first point of the training-data-set.
    """

    def __init__(self
        , max_levels = 5
        , convex_nbd = 5
        , n_neighbors = 5
        , max_concov = None
        , theta = 1.0
        , shadow = 100
        , sigma = 0.000001
        , n_jobs = 1
        , debug = False
        ):
        """
        Initializes the class and mark it as untrained.
        """
        self.data = None
        self.max_levels = max_levels
        self.convex_nbd = convex_nbd
        self.n_neighbors = n_neighbors
        self.max_concov = max_concov
        self.theta = theta
        self.shadow = shadow
        self.sigma = sigma
        self.n_jobs = n_jobs
        self.debug = debug
        self.canPredict = False

    def reset(self, _dataSet):
        """
        Resets the trained GAN to an random state.
        """
        pass

    def train(self, dataSet):
        """
        Trains the GAN.

        It stores the first data-point in the training data-set and mark the GAN as trained.

        *dataSet* is a instance of /library.dataset.DataSet/. It contains the training dataset.
        We are only interested in the class 1.
        """
        self.data = dataSet

    def generateDataPoint(self):
        """
        Generates one synthetic data-point by copying the stored data point.
        """
        return self.generateData(1)[0]

    def generateData(self, numOfSamples=1):
        """
        Generates a list of synthetic data-points.

        *numOfSamples* is a integer > 0. It gives the number of new generated samples.
        """
        if self.max_concov is not None:
            max_concov = self.max_concov
        else:
            max_concov = self.data.data.shape[0]

        return ProWRAS_gen(
            data = self.data.data,
            labels = self.data.labels,
            max_levels = self.max_levels,
            convex_nbd = self.convex_nbd,
            n_neighbors = self.n_neighbors,
            max_concov = max_concov,
            num_samples_to_generate = numOfSamples,
            theta = self.theta,
            shadow = self.shadow,
            sigma = self.sigma,
            n_jobs = self.n_jobs,
            enableDebug = self.debug)[0][:numOfSamples]
