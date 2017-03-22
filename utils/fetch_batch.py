import numpy as np

def fetch_batch(X, y, epoch, n_batches, batch_index, batch_size):
    """
    A generic function that returns the next batch of data to train on.

    Parameters
    ==========

    :X: The training examples
    :y: Target labels
    :epoch: The current iteration. Helps generate reproduceable sequences.
    :n_batches: The number of batches per epoch
    :batch_index: The current batch we're training on.
    :batch_size: The number of training examples per batch.

    Summary
    =======

    In a mini-batch optimization problem we take a small number of random
    examples to perform one step of gradient descent. This is in contrast
    to a batch optimization which uses the entire training set to
    perform one step.

    """
    # Seed the random number generator to reproduce the random sequence
    np.random.seed(epoch * n_batches + batch_index)

    # The number of training examples
    m_examples = len(X)

    # A random sequence of index positions sampled from the training set.
    # The size is depends on how large we want the batch to be. A batch
    # size equal to the m_examples is the same as processing the whole
    # batch in memory at once.
    indices = np.random.randint(m_examples, size=batch_size, dtype=np.int32)

    # Return the examples that correspond to the randomly sampled index positions.
    return (X[indices], y[indices])
