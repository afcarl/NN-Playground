"""


"""

from sklearn import preprocessing

from data import mnist
from lib import cells

# <config>
path = "/home/schlag/MyStuff/Data/MNIST/"

# gaussian initialization variance
stddev = 0.01

number_of_labels = 10

# </config>

train_images, train_labels = [], []
graph = []

def load_data():
	train_images, train_labels = mnist.load_mnist(dataset="training", path=path)
	tl = np.zeros((train_labels.shape[0], number_of_labels))
	for i in range(train_labels.shape[0]):
		tl[i][train_labels[i]] = 1
	train_labels = tl


def build():



