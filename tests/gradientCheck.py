import numpy as np

class GradientCheck:

  def feed_forward_check(self, net, input_size, target_size):

    batch = np.array([
      np.random.rand(input_size),
      np.random.rand(input_size)
    ])

    target = np.array([
      np.zeros(target_size),
      np.zeros(target_size)
    ])
    target[0][np.random.randint(0, target_size)] = 1
    target[1][np.random.randint(0, target_size)] = 1

    net.forward(batch, target)
    net.backward()

    for trainable_element in net.train_objects:
      print("checking element: ", trainable_element.get_name())
      no_error = True
      dW = trainable_element.get_gradient()
      W = trainable_element.get_weights()

      h = 0.000001
      error_threshold = 0.01
      it = np.nditer(W, flags=['multi_index'], op_flags=['readwrite'])

      while not it.finished:
        ix = it.multi_index
        orig = W[ix]

        W[ix] = orig + h

        plus_loss = net.forward(batch, target)[0]

        W[ix] = orig - h
        minus_loss = net.forward(batch, target)[0]

        # calculate the numerical gradient (f(x+h)-f(x-h)) / 2h
        numeric_grad = (plus_loss - minus_loss) / (2 * h)

        W[ix] = orig

        relative_error = []
        for i in range(len(dW)):
          re = np.abs(dW[i][ix] - numeric_grad[i]) \
                         / (np.abs(dW[i][ix]) + np.abs(numeric_grad[i]))
          relative_error.append(re)

        batch_compare = np.asarray(relative_error) > np.asarray([error_threshold])
        if batch_compare.any():
          failed_sample_indexes = np.where(batch_compare == True)[0]
          for si in failed_sample_indexes:
            print("sample: ", si)
            print("gradient index: ", ix)
            print(" numeric gradient: ", numeric_grad[si])
            print("backprop gradient: ", dW[si][ix])
            print("relative error: ", relative_error[si])
            print()
          no_error = False

        it.iternext()
      if no_error:
        print(trainable_element.get_name(), " -> ok")


