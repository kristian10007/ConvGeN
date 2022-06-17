"""
This module contains some example Generative Adversarial Networks for testing.

The classes StupidToyPointGan and StupidToyListGan are not really Networks. This classes are used
for testing the interface. Hope your actually GAN will perform better than this two.

The class SimpleGan is a simple standard Generative Adversarial Network.
"""


import numpy as np

from library.interfaces import GanBaseClass

from keras.layers import Dense, Dropout, Input
from keras.models import Model, Sequential
from keras.layers.advanced_activations import LeakyReLU
from tensorflow.keras.optimizers import Adam
import tensorflow as tf


class SimpleGan(GanBaseClass):
    """
    A class for a simple GAN.
    """
    def __init__(self, numOfFeatures=786, noiseSize=None, epochs=10, batchSize=128, withTanh=False, gLayers=None, dLayers=None):
        self.canPredict = False
        self.isTrained = False
        self.noiseSize = noiseSize if noiseSize is not None else (numOfFeatures * 16)
        self.numOfFeatures = numOfFeatures
        self.epochs = epochs
        self.batchSize = batchSize
        self.scaler = 1.0
        self.withTanh = withTanh
        self.dLayers = dLayers if dLayers is not None else [numOfFeatures * 40, numOfFeatures * 20, numOfFeatures * 10]
        self.gLayers = gLayers if gLayers is not None else [self.noiseSize * 2, numOfFeatures * 4, numOfFeatures * 2]

    def reset(self, _dataSet):
        """
        Resets the trained GAN to an random state.
        """
        self.scaler = 1.0
        self.generator = self._createGenerator(self.numOfFeatures, self.noiseSize)
        self.discriminator = self._createDiscriminator(self.numOfFeatures)
        self.gan = self._createGan(self.noiseSize)

    @staticmethod
    def _adamOptimizer():
        return Adam(learning_rate=0.0002, beta_1=0.5)

    def _createGan(self, noiseSize=100):
        self.discriminator.trainable=False
        gan_input = Input(shape=(noiseSize,))
        x = self.generator(gan_input)
        gan_output = self.discriminator(x)
        gan= Model(inputs=gan_input, outputs=gan_output)
        gan.compile(loss='binary_crossentropy', optimizer='adam')
        return gan

    def _createGenerator(self, numOfFeatures, noiseSize):
        generator=Sequential()
        for (n, size) in enumerate(self.dLayers):
            if n == 0:
                generator.add(Dense(units=size, input_dim=noiseSize))
                generator.add(LeakyReLU(0.2))
            else:
                generator.add(Dense(units=size))
                generator.add(LeakyReLU(0.2))


        if self.withTanh:
            generator.add(Dense(units=numOfFeatures, activation='tanh'))
        else:
            generator.add(Dense(units=numOfFeatures, activation='softsign'))

        generator.compile(loss='binary_crossentropy', optimizer=self._adamOptimizer())
        return generator

    def _createDiscriminator(self, numOfFeatures):
        discriminator=Sequential()

        for (n, size) in enumerate(self.dLayers):
            if n == 0:
                discriminator.add(Dense(units=size, input_dim=numOfFeatures))
                discriminator.add(LeakyReLU(0.2))
            else:
                discriminator.add(Dropout(0.3))
                discriminator.add(Dense(units=size))
                discriminator.add(LeakyReLU(0.2))

        discriminator.add(Dense(units=1, activation='sigmoid'))

        discriminator.compile(loss='binary_crossentropy', optimizer=self._adamOptimizer())
        return discriminator

    def train(self, dataset):
        trainData = dataset.data1
        trainDataSize = trainData.shape[0]

        if trainDataSize <= 0:
            raise AttributeError("Train GAN: Expected data class 1 to contain at least one point.")

        if self.withTanh:
            self.scaler = 1.0
            scaleDown = 1.0
        else:
            self.scaler = max(1.0, 1.1 * tf.reduce_max(tf.abs(trainData)).numpy())
            scaleDown = 1.0 / self.scaler

        trainData = scaleDown * trainData

        for e in range(self.epochs):
            print(f"Epoch {e + 1}/{self.epochs}")
            for _ in range(self.batchSize):
                #generate  random noise as an input  to  initialize the  generator
                noise= np.random.normal(0, 1, [self.batchSize, self.noiseSize])

                # Generate fake MNIST images from noised input
                syntheticBatch = self.generator.predict(noise)

                # Get a random set of  real images
                realBatch = trainData[
                    np.random.randint(low=0, high=trainDataSize, size=self.batchSize)
                    ]

                #Construct different batches of  real and fake data
                X = np.concatenate([realBatch, syntheticBatch])

                # Labels for generated and real data
                y_dis=np.zeros(2 * self.batchSize)
                y_dis[:self.batchSize] = 0.9

                #Pre train discriminator on  fake and real data  before starting the gan.
                self.discriminator.trainable = True
                self.discriminator.train_on_batch(X, y_dis)

                #Tricking the noised input of the Generator as real data
                noise = np.random.normal(0, 1, [self.batchSize, self.noiseSize])
                y_gen = np.ones(self.batchSize)

                # During the training of gan,
                # the weights of discriminator should be fixed.
                #We can enforce that by setting the trainable flag.
                self.discriminator.trainable=False

                #training  the GAN by alternating the training of the Discriminator
                #and training the chained GAN model with Discriminator’s weights freezed.
                self.gan.train_on_batch(noise, y_gen)


    def generateDataPoint(self):
        return self.generateData(1)[0]


    def generateData(self, numOfSamples=1):
        #generate  random noise as an input  to  initialize the  generator
        noise = np.random.normal(0, 1, [numOfSamples, self.noiseSize])

        # Generate fake MNIST images from noised input
        return self.scaler * self.generator.predict(noise)
