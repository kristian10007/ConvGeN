import numpy as np
import matplotlib.pyplot as plt

from library.interfaces import GanBaseClass
from library.dataset import DataSet

from keras.layers import Dense, Input, Multiply, Flatten, Conv1D, Reshape
from keras.models import Model
from keras import backend as K
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Lambda

from sklearn.utils import shuffle

from library.NNSearch import NNSearch

import warnings
warnings.filterwarnings("ignore")



def repeat(x, times):
    return [x for _i in range(times)]

def create01Labels(totalSize, sizeFirstHalf):
    labels = repeat(np.array([1,0]), sizeFirstHalf)
    labels.extend(repeat(np.array([0,1]), totalSize - sizeFirstHalf))
    return np.array(labels)

class ConvGeN(GanBaseClass):
    """
    This is the ConvGeN class. ConvGeN is a synthetic point generator for imbalanced datasets.
    """
    def __init__(self, n_feat, neb=5, gen=None, neb_epochs=10, maj_proximal=False, debug=False):
        self.isTrained = False
        self.n_feat = n_feat
        self.neb = neb
        self.nebInitial = neb
        self.genInitial = gen
        self.gen = gen if gen is not None else self.neb
        self.neb_epochs = neb_epochs
        self.loss_history = None
        self.debug = debug
        self.minSetSize = 0
        self.conv_sample_generator = None
        self.maj_min_discriminator = None
        self.maj_proximal = maj_proximal
        self.cg = None
        self.canPredict = True

        if self.neb is not None and self.gen is not None and self.neb > self.gen:
            raise ValueError(f"Expected neb <= gen but got neb={neb} and gen={gen}.")

    def reset(self, dataSet):
        """
        Creates the network.

        *dataSet* is a instance of /library.dataset.DataSet/ or None.
        It contains the training dataset.
        It is used to determine the neighbourhood size if /neb/ in /__init__/ was None.
        """
        self.isTrained = False

        if dataSet is not None:
            nMinoryPoints = dataSet.data1.shape[0]
            if self.nebInitial is None:
                self.neb = nMinoryPoints
            else:
                self.neb = min(self.nebInitial, nMinoryPoints)
        else:
            self.neb = self.nebInitial

        self.gen = self.genInitial if self.genInitial is not None else self.neb

        ## instanciate generator network and visualize architecture
        self.conv_sample_generator = self._conv_sample_gen()

        ## instanciate discriminator network and visualize architecture
        self.maj_min_discriminator = self._maj_min_disc()

        ## instanciate network and visualize architecture
        self.cg = self._convGeN(self.conv_sample_generator, self.maj_min_discriminator)

        if self.debug:
            print(f"neb={self.neb}, gen={self.gen}")

            print(self.conv_sample_generator.summary())
            print('\n')
            
            print(self.maj_min_discriminator.summary())
            print('\n')

            print(self.cg.summary())
            print('\n')

    def train(self, dataSet, discTrainCount=5):
        """
        Trains the Network.

        *dataSet* is a instance of /library.dataset.DataSet/. It contains the training dataset.
        
        *discTrainCount* gives the number of extra training for the discriminator for each epoch. (>= 0)
        """
        if dataSet.data1.shape[0] <= 0:
            raise AttributeError("Train: Expected data class 1 to contain at least one point.")

        # Store size of minority class. This is needed during point generation.
        self.minSetSize = dataSet.data1.shape[0]

        # Precalculate neighborhoods
        self.nmbMin = NNSearch(self.neb).fit(haystack=dataSet.data1)
        if self.maj_proximal:
            self.nmbMaj = NNSearch(self.neb).fit(haystack=dataSet.data0, needles=dataSet.data1)
        else:
            self.nmbMaj = None

        # Do the training.
        self._rough_learning(dataSet.data1, dataSet.data0, discTrainCount)
        
        # Neighborhood in majority class is no longer needed. So save memory.
        self.nmbMaj = None
        self.isTrained = True

    def generateDataPoint(self):
        """
        Returns one synthetic data point by repeating the stored list.
        """
        return (self.generateData(1))[0]


    def generateData(self, numOfSamples=1):
        """
        Generates a list of synthetic data-points.

        *numOfSamples* is a integer > 0. It gives the number of new generated samples.
        """
        if not self.isTrained:
            raise ValueError("Try to generate data with untrained network.")

        ## roughly claculate the upper bound of the synthetic samples to be generated from each neighbourhood
        synth_num = (numOfSamples // self.minSetSize) + 1

        ## generate synth_num synthetic samples from each minority neighbourhood
        synth_set=[]
        for i in range(self.minSetSize):
            synth_set.extend(self._generate_data_for_min_point(i, synth_num))

        ## extract the exact number of synthetic samples needed to exactly balance the two classes
        synth_set = np.array(synth_set[:numOfSamples]) 

        return synth_set

    def predictReal(self, data):
        """
        Uses the discriminator on data.
        
        *data* is a numpy array of shape (n, n_feat) where n is the number of datapoints and n_feat the number of features.
        """
        prediction = self.maj_min_discriminator.predict(data)
        return np.array([x[0] for x in prediction])

    # ###############################################################
    # Hidden internal functions
    # ###############################################################

    # Creating the Network: Generator
    def _conv_sample_gen(self):
        """
        The generator network to generate synthetic samples from the convex space
        of arbitrary minority neighbourhoods
        """

        ## takes minority batch as input
        min_neb_batch = Input(shape=(self.n_feat,))

        ## reshaping the 2D tensor to 3D for using 1-D convolution,
        ## otherwise 1-D convolution won't work.
        x = tf.reshape(min_neb_batch, (1, self.neb, self.n_feat), name=None)
        ## using 1-D convolution, feature dimension remains the same
        x = Conv1D(self.n_feat, 3, activation='relu')(x)
        ## flatten after convolution
        x = Flatten()(x)
        ## add dense layer to transform the vector to a convenient dimension
        x = Dense(self.neb * self.gen, activation='relu')(x)

        ## again, witching to 2-D tensor once we have the convenient shape
        x = Reshape((self.neb, self.gen))(x)
        ## column wise sum
        s = K.sum(x, axis=1)
        ## adding a small constant to always ensure the column sums are non zero.
        ## if this is not done then during initialization the sum can be zero.
        s_non_zero = Lambda(lambda x: x + .000001)(s)
        ## reprocals of the approximated column sum
        sinv = tf.math.reciprocal(s_non_zero)
        ## At this step we ensure that column sum is 1 for every row in x.
        ## That means, each column is set of convex co-efficient
        x = Multiply()([sinv, x])
        ## Now we transpose the matrix. So each row is now a set of convex coefficients
        aff=tf.transpose(x[0])
        ## We now do matrix multiplication of the affine combinations with the original
        ## minority batch taken as input. This generates a convex transformation
        ## of the input minority batch
        synth=tf.matmul(aff, min_neb_batch)
        ## finally we compile the generator with an arbitrary minortiy neighbourhood batch
        ## as input and a covex space transformation of the same number of samples as output
        model = Model(inputs=min_neb_batch, outputs=synth)
        opt = Adam(learning_rate=0.001)
        model.compile(loss='mean_squared_logarithmic_error', optimizer=opt)
        return model

    # Creating the Network: discriminator
    def _maj_min_disc(self):
        """
        the discriminator is trained in two phase:
        first phase:  while training ConvGeN the discriminator learns to differentiate synthetic
                      minority samples generated from convex minority data space against
                      the borderline majority samples
        second phase: after the ConvGeN generator learns to create synthetic samples,
                      it can be used to generate synthetic samples to balance the dataset
                      and then rettrain the discriminator with the balanced dataset
        """

        ## takes as input synthetic sample generated as input stacked upon a batch of
        ## borderline majority samples
        samples = Input(shape=(self.n_feat,))
        
        ## passed through two dense layers
        y = Dense(250, activation='relu')(samples)
        y = Dense(125, activation='relu')(y)
        y = Dense(75, activation='relu')(y)
        
        ## two output nodes. outputs have to be one-hot coded (see labels variable before)
        output = Dense(2, activation='sigmoid')(y)
        
        ## compile model
        model = Model(inputs=samples, outputs=output)
        opt = Adam(learning_rate=0.0001)
        model.compile(loss='binary_crossentropy', optimizer=opt)
        return model

    # Creating the Network: ConvGeN
    def _convGeN(self, generator, discriminator):
        """
        for joining the generator and the discriminator
        conv_coeff_generator-> generator network instance
        maj_min_discriminator -> discriminator network instance
        """
        ## by default the discriminator trainability is switched off.
        ## Thus training ConvGeN means training the generator network as per previously
        ## trained discriminator network.
        discriminator.trainable = False

        ## input receives a neighbourhood minority batch
        ## and a proximal majority batch concatenated
        batch_data = Input(shape=(self.n_feat,))
        
        ## extract minority batch
        min_batch = Lambda(lambda x: x[:self.neb])(batch_data)
        
        ## extract majority batch
        maj_batch = Lambda(lambda x: x[self.gen:])(batch_data)
        
        ## pass minority batch into generator to obtain convex space transformation
        ## (synthetic samples) of the minority neighbourhood input batch
        conv_samples = generator(min_batch)
        
        ## concatenate the synthetic samples with the majority samples
        new_samples = tf.concat([conv_samples, maj_batch],axis=0)
        
        ## pass the concatenated vector into the discriminator to know its decisions
        output = discriminator(new_samples)
        
        ## note that, the discriminator will not be traied but will make decisions based
        ## on its previous training while using this function
        model = Model(inputs=batch_data, outputs=output)
        opt = Adam(learning_rate=0.0001)
        model.compile(loss='mse', optimizer=opt)
        return model

    # Create synthetic points
    def _generate_data_for_min_point(self, index, synth_num):
        """
        generate synth_num synthetic points for a particular minoity sample
        synth_num -> required number of data points that can be generated from a neighbourhood
        data_min -> minority class data
        neb -> oversampling neighbourhood
        index -> index of the minority sample in a training data whose neighbourhood we want to obtain
        """

        runs = int(synth_num / self.neb) + 1
        synth_set = []
        for _run in range(runs):
            batch = self.nmbMin.getNbhPointsOfItem(index)
            synth_batch = self.conv_sample_generator.predict(batch, batch_size=self.neb)
            synth_set.extend(synth_batch)

        return synth_set[:synth_num]



    # Training
    def _rough_learning(self, data_min, data_maj, discTrainCount):
        generator = self.conv_sample_generator
        discriminator = self.maj_min_discriminator
        convGeN = self.cg
        loss_history = [] ## this is for stroring the loss for every run
        step = 0
        minSetSize = len(data_min)

        labels = tf.convert_to_tensor(create01Labels(2 * self.gen, self.gen))
        nLabels = 2 * self.gen

        for neb_epoch_count in range(self.neb_epochs):
            if discTrainCount > 0:
                for n in range(discTrainCount):
                    for min_idx in range(minSetSize):
                        ## generate minority neighbourhood batch for every minority class sampls by index
                        min_batch_indices = shuffle(self.nmbMin.neighbourhoodOfItem(min_idx))
                        min_batch = self.nmbMin.getPointsFromIndices(min_batch_indices)
                        ## generate random proximal majority batch
                        maj_batch = self._BMB(data_maj, min_batch_indices)

                        ## generate synthetic samples from convex space
                        ## of minority neighbourhood batch using generator
                        conv_samples = generator.predict(min_batch, batch_size=self.neb)
                        ## concatenate them with the majority batch
                        concat_sample = tf.concat([conv_samples, maj_batch], axis=0)

                        ## switch on discriminator training
                        discriminator.trainable = True
                        ## train the discriminator with the concatenated samples and the one-hot encoded labels
                        discriminator.fit(x=concat_sample, y=labels, verbose=0, batch_size=20)
                        ## switch off the discriminator training again
                        discriminator.trainable = False

            for min_idx in range(minSetSize):
                ## generate minority neighbourhood batch for every minority class sampls by index
                min_batch_indices = shuffle(self.nmbMin.neighbourhoodOfItem(min_idx))
                min_batch = self.nmbMin.getPointsFromIndices(min_batch_indices)
                
                ## generate random proximal majority batch
                maj_batch = self._BMB(data_maj, min_batch_indices)

                ## generate synthetic samples from convex space
                ## of minority neighbourhood batch using generator
                conv_samples = generator.predict(min_batch, batch_size=self.neb)
                
                ## concatenate them with the majority batch
                concat_sample = tf.concat([conv_samples, maj_batch], axis=0)

                ## switch on discriminator training
                discriminator.trainable = True
                ## train the discriminator with the concatenated samples and the one-hot encoded labels
                discriminator.fit(x=concat_sample, y=labels, verbose=0, batch_size=20)
                ## switch off the discriminator training again
                discriminator.trainable = False

                ## use the complete network to make the generator learn on the decisions
                ## made by the previous discriminator training
                gen_loss_history = convGeN.fit(concat_sample, y=labels, verbose=0, batch_size=nLabels)

                ## store the loss for the step
                loss_history.append(gen_loss_history.history['loss'])

                step += 1
                if self.debug and (step % 10 == 0):
                    print(f"{step} neighbourhood batches trained; running neighbourhood epoch {neb_epoch_count}")

            if self.debug:
                print(f"Neighbourhood epoch {neb_epoch_count + 1} complete")

        if self.debug:
            run_range = range(1, len(loss_history) + 1)
            plt.rcParams["figure.figsize"] = (16,10)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.xlabel('runs', fontsize=25)
            plt.ylabel('loss', fontsize=25)
            plt.title('Rough learning loss for discriminator', fontsize=25)
            plt.plot(run_range, loss_history)
            plt.show()

        self.conv_sample_generator = generator
        self.maj_min_discriminator = discriminator
        self.cg = convGeN
        self.loss_history = loss_history


    def _BMB(self, data_maj, min_idxs):

        ## Generate a borderline majority batch
        ## data_maj -> majority class data
        ## min_idxs -> indices of points in minority class
        ## gen -> convex combinations generated from each neighbourhood

        if self.nmbMaj is not None:
            return self.nmbMaj.neighbourhoodOfItemList(shuffle(min_idxs), maxCount=self.gen)
        else:
            return tf.convert_to_tensor(data_maj[np.random.randint(len(data_maj), size=self.gen)])


    def retrainDiscriminitor(self, data, labels):
        self.maj_min_discriminator.trainable = True
        labels = np.array([ [x, 1 - x] for x in labels])
        self.maj_min_discriminator.fit(x=data, y=labels, batch_size=20, epochs=self.neb_epochs)
        self.maj_min_discriminator.trainable = False
