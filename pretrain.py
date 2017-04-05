# The challenge is to build a deep neural network with:
#
#       => 5 hidden layers
#       => 100 neurons per hidden layer
#       => Weights are initialized using He initialization
#       => Each layer uses the ELU activation function
#       => Use Adam optimization
#       => Implement early stopping
#       => Output layer should use softmax with 5 output neurons
#
#
# Train the network on the MNIST dataset on digits 0 to 4.

# NOTE
# In order to monitor GPU usage in real time, use the
# watch command with the arguments below:
#
#       watch -n 5 nvidia-smi -a --display=utilization
#

import os
from datetime import datetime
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.layers           import fully_connected
from tensorflow.contrib.layers           import batch_norm
from tensorflow.contrib.layers           import dropout
from tensorflow.contrib.framework        import arg_scope

from utils.fetch_batch import *
from utils.make_dir import *

##################
# DIRECTORY SETUP
##################
import yaml

file = open("config.yaml", "r")
config = yaml.load(file)
file.close

# Setup log directory for Tensorboard to read from
now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_log_dir = config['log_root']
log_dir = "{}/run-{}".format(root_log_dir, now)
make_dir(log_dir)

# Save the model at the end of training here:
results_dir = config['results_dir']
make_dir(results_dir)

# Save the best model during training here:
winner_dir = config['winners_dir']
make_dir(winner_dir)

#############
# MNIST DATA
#############

mnist = input_data.read_data_sets("./data/")

X_train = mnist.train.images
y_train = mnist.train.labels.astype(np.int32)

X_val = mnist.validation.images
y_val = mnist.validation.labels.astype(np.int32)

# In this version we only want to train the network on digits from 0 to 4.
lt_five_train_idx = [idx for idx, val in enumerate(y_train) if val < 5]
lt_five_val_idx   = [idx for idx, val in enumerate(y_val)   if val < 5]

# Update training set and labels
X_train = X_train[lt_five_train_idx]
y_train = y_train[lt_five_train_idx]

# Update test set and labels
X_val = X_val[lt_five_val_idx]
y_val = y_val[lt_five_val_idx]


######################
# CONSTRUCTION PHASE
######################

# Clear the default graph stack and reset global default graph
tf.reset_default_graph()

m_examples, n_features = X_train.shape
n_neurons = 100
n_outputs = 5 # [0,1,2,3,4]
alpha = 0.01

X = tf.placeholder(tf.float32, shape=(None, n_features), name="X")
y = tf.placeholder(tf.int32, shape=(None), name="y")

# Pass this flag into the feed dict when running batch normalization
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
        # normalizer_fn=batch_norm,
        # normalizer_params=batch_norm_params,
        activation_fn=tf.nn.elu,
        weights_initializer=he_init):

        # Our "deep" network. We can add dropout if we need to by passing
        # a tensor into the dropout function and specifying the probability
        # certain neurons will be dropped:
        #
        #       X_drop = dropout(X, 0.5, is_training=is_training)
        #
        # Then we can pass the dropout layer into the next fully conncted
        # layer in the network.
        #
        hidden1 = fully_connected(X,       n_neurons, scope="hidden1")
        hidden2 = fully_connected(hidden1, n_neurons, scope="hidden2")
        hidden3 = fully_connected(hidden2, n_neurons, scope="hidden3")
        hidden4 = fully_connected(hidden3, n_neurons, scope="hidden4")
        hidden5 = fully_connected(hidden4, n_neurons, scope="hidden5")

        # The output layer (i.e. "logits") will be an (m x 5) dimensional matrix
        # where each instance represents the output neurons of each training
        # example after it has passed through the network.
        #
        # Here the first dimension, m, is the number of training examples
        # in the batch. The second dimension is the number of output labels.
        #
        # In first row of the example below we can see that the network gave
        # the strongest signal to index position 3, or digit 3.
        #
        # array([
        #    [ -4.56217909,  -0.13691133,  -0.99876124,  12.67642975, -0.55929732],
        #    [ -5.32831955,  -0.23710972,  10.90992451,  -0.27201548, -2.65590262],
        #    [ -4.08373165,  -3.34327364,  -3.24882412,  -2.61436868, 11.48922825],
        #    [  0.41864291,  -9.03825283,   0.4146657 ,  -6.59523439, 8.21064186],
        #    [ -2.35521626,  -0.37815359,   9.7282629 ,   0.83002365, -2.85143971],
        #    ...
        #    ...
        #    ...
        # ])
        logits = fully_connected(hidden5, n_outputs, activation_fn=None, scope="output")


with tf.name_scope("loss"):
    # tf.nn.sparse_softmax_cross_entropy_with_logits() expects labels in the form of
    # integers ranging from 0 to the number of classes minus 1. This will
    # return a 1D tensor containing the cross entropy for each instance.
    entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(entropy, name="loss")

with tf.name_scope("train"):
    # The AdamOptimzier is a very good default choice for an optimization
    # strategy. It's much faster than gradient descent at converging.
    optimizer = tf.train.AdamOptimizer()
    training_op = optimizer.minimize(loss)


with tf.name_scope("eval"):
    # Score the model's output as correct if the True label can be found
    # in the K-most-likely predictions. In this case we're setting K
    # to equal 1 so we only consider a prediction correct if it is for
    # the true label. Returns an (m x 5) dimensional matrix of prediction
    # values.
    correct = tf.nn.in_top_k(logits, y, 1)

    # tf.argmax() is going to return the index of the largest value
    # across axes of a tensor. This has the effect of giving us a
    # list of the digits we've predicted. The values we want to compare
    # are stored in the second dimension, so we need to set the axis
    # to 1 (0 represents the first dimension).
    preds = tf.argmax(logits, axis=1)

    # What percent of the predictions are correct?
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

###################
# EXECUTION PHASE
###################

init = tf.global_variables_initializer()
saver = tf.train.Saver()

n_epochs   = 20
batch_size = 50
n_batches  = int(np.ceil(m_examples / batch_size))

###############
# TENSORBOARD
###############

saver = tf.train.Saver()
train_loss_summary = tf.summary.scalar("train xentropy", loss)
val_loss_summary   = tf.summary.scalar("validation xentropy", loss)
summary_writer     = tf.summary.FileWriter(log_dir, tf.get_default_graph())

##################
# EARLY STOPPING
##################
#
# The basic idea is to interrupt training when its performance on
# the validation set starts dropping.
#
# One way to implement this is to evaluate the model on a validation
# set at regular intervals and save a "winner" snapshot if it outperforms
# the previous winner. Do this by counting the number of steps since
# the last winner was saved, and interrupt training when this number
# reaches some limit. Then restore the last winner snapshot.
best_loss  = float(0)
best_acc   = float(0)
best_epoch = int(0)
best_model = None # Will be path to .ckpt file
steps_since_best_epoch = int(0)

# Run the graph...
with tf.Session() as sess:
    init.run()

    col_headers = ["Epoch", "Tr acc", "Val acc", "Tr loss", "Val loss", "Best acc", "Decay"]
    print()
    print("{:^6} {:<10} {:<10} {:<10} {:<10} {:<10} {:^6}".format(*col_headers))
    print("=============================================================================")

    for epoch in range(n_epochs):

        # Should we restore our best model?
        if steps_since_best_epoch > 5:
            saver.restore(sess, best_model)
            restored = True
        else:
            restored = False

        # Iterate through the batches
        for batch_idx in range(m_examples // batch_size):
            # Find the next batch for the current epoch
            X_batch, y_batch = fetch_batch(
                X_train, y_train, epoch, n_batches, batch_idx, batch_size)

            # Run the graph
            sess.run(training_op, feed_dict={is_training: True, X: X_batch, y: y_batch})

            # if batch_idx % 10 == 0:
            #     # Log the results to TensorBoard. We're going to watch the
            #     # loss function on both the training and validation set.
            #     # If we log the results too much it's going to slow down
            #     # training significantly.
            #     step = epoch * n_batches + batch_idx
            #
            #     train_summary = train_loss_summary.eval(feed_dict={X: X_batch, y: y_batch})
            #     val_summary   = val_loss_summary.eval(feed_dict={X: X_val, y: y_val})
            #
            #     summary_writer.add_summary(train_summary, step)
            #     summary_writer.add_summary(val_summary, step)
            #

        # Check how we're doing after each epoch
        acc_train = accuracy.eval(feed_dict={is_training: False, X: X_batch, y: y_batch})
        acc_val   = accuracy.eval(feed_dict={is_training: False, X: X_val, y: y_val})

        loss_train = loss.eval(feed_dict={is_training: False, X: X_batch, y: y_batch})
        loss_val   = loss.eval(feed_dict={is_training: False, X: X_val, y: y_val})

        # Update our best model up to this point. If we've made
        # an improvement, then save it as the best model. Otherwise
        # increment the number of steps it's been since we've had
        # a winner model.
        if acc_val > best_acc:
            best_acc   = acc_val
            best_epoch = epoch
            best_model = saver.save(sess, winner_dir + "/v1_winner.ckpt")
            steps_since_best_epoch = int(0)
        else:
            steps_since_best_epoch += 1

        # We also want to keep track of the best loss function value
        if loss_val < best_loss:
            best_loss = loss_val


        print("{:^6} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:^6}".format(
            epoch, acc_train, acc_val, loss_train, loss_val, best_acc, steps_since_best_epoch))

    # We're done with training. Let's save our final model. Remember that
    # our final model might not be our best model.
    save_path = saver.save(sess, results_dir + "/v1_final.ckpt")

    ####################
    # LEARNING CURVES
    ####################
    #
    # Since TensorBoard cannot plot learning curves, we
    # can plot them using matplotlib. We just need
    # to collect the training and validation errors and
    # the number of optimization steps.
    # print()
    # print("Generating learning curves...")
    #
    # train_errors = []
    # val_errors = []
    #
    # for m in range(1, len(X_val)):
    #     sess.run(training_op, feed_dict={is_training: True, X: X_train[:m], y: y_train[:m]})
    #     train_errors.append(loss.eval(feed_dict={is_training: False, X: X_train[:m], y: y_train[:m]}))
    #     val_errors.append(loss.eval(feed_dict={is_training: False, X: X_val[:m], y: y_val[:m]}))
    #
    # print("Plotting learning curves...")
    # plt.plot(train_errors, "r-+", linewidth=2, label="train")
    # plt.plot(val_errors, "b-", linewidth=3, label="val")
    # plt.legend(loc="upper right", fontsize=14)
    # plt.xlabel("Training set size", fontsize=4)
    # plt.ylabel("XEntropy", fontsize=4)
    # plt.axis([0, len(X_val), 0, best_loss * 10])
    # plt.savefig("images/v1_learning_curves.png")
    # # plt.show()


    #########################
    # EVALUATE THE TEST SET
    #########################

    print("Evalutaing the test set...")
    from sklearn.metrics import precision_score, recall_score, f1_score

    X_test = mnist.test.images
    y_test = mnist.test.labels

    lt_five_test_idx  = [idx for idx, val in enumerate(y_test)  if val < 5]

    X_test = X_test[lt_five_test_idx]
    y_test = y_test[lt_five_test_idx]

    y_pred   = preds.eval(feed_dict={is_training: False, X: X_test, y: y_test})
    conf_mat = tf.confusion_matrix(y_test, y_pred).eval()

    # A better way to evaluate the performance of a classifier is to look at
    # the confusion matrix. The basic idea is to count the number of times
    # instances of class A are classified as class B. For example, to know
    # the number of times the classifier confused images of 5s with 3s,
    # you would look in the 5th row and 3rd column of the confusion matrix.
    #
    # Each row in a confusion matrix represents an actual class, while each
    # column represents a predicted class. Here's an example:
    #
    #
    #            [[478   0   0   1   0]
    #             [  0 559   1   1   2]
    #             [  0   2 484   1   1]
    #             [  1   0   1 490   1]
    #             [  0   0   1   1 533]]
    #
    #
    print()
    print("Confusion Matrix")
    print(conf_mat)
    print()

    # The precision is the ratio tp / (tp + fp) where tp is the number of true
    # positives and fp the number of false positives. The precision is
    # intuitively the ability of the classifier not to label as positive a
    # sample that is negative.
    precision_scores = precision_score(y_test, y_pred, average=None)

    # The recall is the ratio tp / (tp + fn) where tp is the number of true
    # positives and fn the number of false negatives. The recall is intuitively
    # the ability of the classifier to find all the positive samples.
    recall_scores = recall_score(y_test, y_pred, average=None)

    # The F1 score can be interpreted as a weighted average of the precision
    # and recall, where an F1 score reaches its best value at 1 and worst score
    # at 0. The relative contribution of precision and recall to the F1 score
    # are equal. The formula for the F1 score is:
    #
    #       F1 = 2 * (precision * recall) / (precision + recall)
    #
    f1_scores = f1_score(y_test, y_pred, average=None)


    col_headers = ["Digit", "Precision", "Recall", "F1"]
    labels      = [0,1,2,3,4]

    print("{:^12} {:^12} {:^12} {:^12}".format(*col_headers))
    print("===================================================================")

    for idx, val in enumerate(labels):
        print("{:^12} {:^12.4f} {:^12.4f} {:^12.4f}".format(
            labels[idx], precision_scores[idx], recall_scores[idx], f1_scores[idx]))


    print("Session finished! :)")
