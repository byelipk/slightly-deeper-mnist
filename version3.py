# The challenge is to build a deep neural network with:
#
#       => 5 hidden layers
#       => 100 neurons per hidden layer
#       => Weights are initialized using He initialization
#       => Each layer uses the ELU activation function
#       => Use Adam optimization
#       => Implement early stopping
#       => Output layer should use softmax with 5 output neurons
#       => Enable batch normalization
#       => Add dropout to every layer
#
#
# Train the network on the MNIST dataset on digits 0 to 4.

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.layers           import fully_connected
from tensorflow.contrib.layers           import batch_norm
from tensorflow.contrib.layers           import dropout
from tensorflow.contrib.framework        import arg_scope

from utils.fetch_batch import *


mnist = input_data.read_data_sets("./data/")

X_train = mnist.train.images
y_train = mnist.train.labels.astype(np.int32)

X_val = mnist.validation.images
y_val = mnist.validation.labels.astype(np.int32)

# In this version we only want to train the network on digits from 0 to 4.
lt_five_train_idx = [idx for idx, val in enumerate(y_train) if val < 5]
lt_five_val_idx  = [idx for idx, val in enumerate(y_val)  if val < 5]

# Update training set and labels
X_train = X_train[lt_five_train_idx]
y_train = y_train[lt_five_train_idx]

# Update test set and labels
X_val = X_val[lt_five_val_idx]
y_val = y_val[lt_five_val_idx]

# CONSTRUCTION PHASE
#
#
#

# Clear the default graph stack and reset global default graph
tf.reset_default_graph()

m_examples, n_features = X_train.shape
n_neurons = 100
n_outputs = 5 # [0,1,2,3,4]
alpha = 0.01

X = tf.placeholder(tf.float32, shape=(None, n_features), name="X")
y = tf.placeholder(tf.int32, shape=(None), name="y")
is_training = tf.placeholder(tf.bool, shape=(), name="is_training")

with tf.name_scope("DNN"):
    # The initialization strategy for the ELU, ReLU and its variants
    # is called "He initialization". This helps alleviate the vanishing
    # gradients problem.
    he_init = tf.contrib.layers.variance_scaling_initializer()

    # Batch normalization lets the model learn the optimal scale
    # and mean of the inputs for each layer and significantly
    # reduces the vanishing gradients problem.
    batch_norm_params = {
        "is_training": is_training,
        "decay": 0.9,
        "updates_collections": None,
        "scale": None
    }

    # These arguments are applied to `fully_connected()` each time
    # it is invoked. It helps clean up our code.
    with arg_scope(
        [fully_connected],
        activation_fn=tf.nn.elu,
        weights_initializer=he_init,
        normalizer_fn=batch_norm,
        normalizer_params=batch_norm_params):

        X_drop = dropout(X, 0.5, is_training=is_training)

        # Our "deep" network
        hidden1 = fully_connected(X_drop,  n_neurons, scope="hidden1")
        drop1   = dropout(hidden1, 0.5, is_training=is_training)

        hidden2 = fully_connected(drop1, n_neurons, scope="hidden2")
        drop2   = dropout(hidden2, 0.5, is_training=is_training)

        hidden3 = fully_connected(drop2, n_neurons, scope="hidden3")
        drop3   = dropout(hidden3, 0.5, is_training=is_training)

        hidden4 = fully_connected(drop3, n_neurons, scope="hidden4")
        drop4   = dropout(hidden4, 0.5, is_training=is_training)

        hidden5 = fully_connected(drop4, n_neurons, scope="hidden5")
        drop5   = dropout(hidden5, 0.5, is_training=is_training)

        # The output layer
        logits = fully_connected(drop5, n_outputs, activation_fn=None, scope="output")


with tf.name_scope("loss"):
    # tf.nn.sparse_softmax_cross_entropy_with_logits() expects labels in the form of
    # integers ranging from 0 to the number of classes minus 1. This will
    # return a 1D tensor containing the cross entropy for each instance.
    entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(entropy, name="loss")

with tf.name_scope("train"):
    optimizer = tf.train.AdamOptimizer()
    training_op = optimizer.minimize(loss)


with tf.name_scope("eval"):
    # For each instance determine if the highest logit corresponds to the
    # target class. Returns a 1D tensor of boolean values.
    correct = tf.nn.in_top_k(logits, y, 1)

    # What percent of the predictions are correct?
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    # TODO: What percent of the positive cases did we catch?
    # TODO: What percent of positive predictions were correct?

# EXECUTION PHASE
#
#
#

init = tf.global_variables_initializer()
saver = tf.train.Saver()

n_epochs   = 20
batch_size = 50
n_batches  = m_examples // batch_size

# Early stopping
#
# The basic idea is to interrupt training when its performance on
# the validation set starts dropping.
#
# One way to implement this is to evaluate the model on a validation
# set at regular intervals and save a "winner" snapshot if it outperforms
# the previous winner. Do this by counting the number of steps since
# the last winner was saved, and interrupt training when this number
# reaches some limit. Then restore the last winner snapshot.
best_acc   = float(0)
best_epoch = int(0)
best_model = None # Will be path to .ckpt file
steps_since_best_epoch = int(0)

with tf.Session() as sess:
    init.run()

    col_headers = ["Epoch", "Train acc.", "Val acc.", "Best acc.", "Decay", "Restored"]
    print("{:^6} {:^10} {:^10} {:^10} {:^6} {:^8}".format(*col_headers))
    print("===================================================================")

    for epoch in range(n_epochs):

        # Should we restore our best model?
        if epoch % 2 == 0 and steps_since_best_epoch > 3:
            saver.restore(sess, best_model)
            restored = True
        else:
            restored = False

        # Iterate through the batches
        for batch_idx in range(m_examples // batch_size):
            X_batch, y_batch = fetch_batch(
                X_train, y_train, epoch, n_batches, batch_idx, batch_size)
            sess.run(training_op, feed_dict={is_training: True, X: X_batch, y: y_batch})

        acc_train = accuracy.eval(feed_dict={is_training: False, X: X_batch, y: y_batch})
        acc_val  = accuracy.eval(feed_dict={is_training: False, X: X_val, y: y_val})

        # Update our best
        if acc_val > best_acc:
            best_acc   = acc_val
            best_epoch = epoch
            best_model = saver.save(sess, "winners/v3_winner.ckpt")
            steps_since_best_epoch = int(0)
        else:
            # Keep track of how long it's been since we've had a "winner" model
            steps_since_best_epoch += 1

        print("{:^6} {:<10.4f} {:<10.4f} {:<10.4f} {:^6} {:^8}".format(
            epoch, acc_train, acc_val, best_acc, steps_since_best_epoch, restored))

    save_path = saver.save(sess, "results/v3_final.ckpt")
