import numpy as np

from abc import ABCMeta, abstractmethod


class GraphOperation(metaclass=ABCMeta):
  """
  Abstract base class for all operations performed on this graph.
  """

  @abstractmethod
  def forward(self, x):
    pass

  @abstractmethod
  def backward(self, delta):
    pass


class Graph:
  """
  Performs operations on the computational graph.
  """
  graph = []

  def add(self, operation):
    """
    Adds GraphOperations to the end of the "graph".
    :param operation: An implementation of the abstract GraphOperation base class
    :return: None
    """
    assert isinstance(operation, GraphOperation)
    self.graph.append(operation)

  def add_list(self, list_of_operations):
    """
    Checks if all operations in the list are GraphOperations and adds each using add().
    :param list_of_operations: list of GraphOperation implementations
    :return: None
    """
    for op in list_of_operations:
      assert isinstance(op, GraphOperation)

    for op in list_of_operations:
      self.add(op)

  def forward(self, x):
    """
    Perform a forward pass through the computational graph.
    :param x: Input data rows: samples cols: sample dimensions
    :return:
    """
    t = x.copy()
    for op in self.graph:
      t = op.forward(t)
    return t