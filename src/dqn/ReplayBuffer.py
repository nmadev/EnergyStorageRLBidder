from collections import deque
import random

class ReplayBuffer:
    def __init__(self, maxlength):
        """
        maxlength: Max number of tuples to store in the buffer.
        """
        self.buffer = deque(maxlen=maxlength)
        self.maxlength = maxlength

    def append(self, experience):
        """
        Appends a new experience tuple to the buffer.
        experience: A tuple of the form (state, action, reward, next_state).
        """
        self.buffer.append(experience)

    def sample(self, batchsize):
        """
        Samples 'batchsize' experience tuples from the buffer.
        batchsize: Size of the minibatch to be sampled.
        return: A list of tuples of form (state, action, reward, next_state)
        """
        minibatch = random.sample(self.buffer, batchsize)
        return minibatch
