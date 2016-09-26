"""
2D visualization test network
"""

import numpy as np
from misc.utils import shuffle_in_unison
from networks.SimpleLinearNet import SimpleLinearNet
import trainers.sgd

# generate X data for red and blue class (two gaussian)
red_x1 = np.random.normal(-0.5, 0.2, 100)
red_x2 = np.random.normal(-0.5, 0.2, 100)
red = np.dstack((red_x1, red_x2))[0]

blue_x1 = np.random.normal(0.5, 0.2, 100)
blue_x2 = np.random.normal(0.5, 0.2, 100)
blue = np.dstack((blue_x1, blue_x2))[0]

X = np.concatenate((red, blue), axis=0)

# targets
red_target = np.dstack((np.ones(red.shape[0]), np.zeros(red.shape[0])))[0]
blue_target = np.dstack((np.zeros(blue.shape[0]), np.ones(blue.shape[0])))[0]

Y = np.concatenate((red_target, blue_target), axis=0)

#shuffle data
shuffle_in_unison(X, Y)

net = SimpleLinearNet()
sgd = trainers.sgd.SGD()


sgd.train(X, Y, epochs=5, train_batch_size=20, eval_batch_size=200, learning_rate=0.01, network=net)

