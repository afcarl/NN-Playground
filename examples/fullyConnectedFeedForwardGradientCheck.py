"""
Perform numerical gradient checking on a FC MNIST network.
"""

from networks.fcMnistGradientCheck import FullyConnectedFeedForward
from tests.gradientCheck import GradientCheck

net = FullyConnectedFeedForward()
gc = GradientCheck()

gc.feed_forward_check(net, 20, 20)
