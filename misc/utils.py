import numpy as np

def shuffle_in_unison(a, b):
  """
  Shuffles two np arrays in the same way.
  :param a: first array
  :param b: second array
  :return: none
  """
  rng_state = np.random.get_state()
  np.random.shuffle(a)
  np.random.set_state(rng_state)
  np.random.shuffle(b)