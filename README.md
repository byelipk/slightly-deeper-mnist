### Deep Learning

* Build a DNN with five hidden layers of 100 neurons each, using He initialization for the weights and the ELU activation function.
* Using Adam optimization and early stopping, try training MNIST but only on digits 0 to 4. (We will use transfer learning for digits 5 to 9 in the next exercise.) You will need a softmax output layer with 5 neurons, and as always make sure to save checkpoints at regular intervals and save the final model so you can reuse it later.
* Tune the hyperparameters using cross-validation and see what precision you can achieve.
* Now try adding batch normalization and compare the learning curves: is it converging faster than before? Does it produce a better model?
* Is the model overfitting the training set? Try adding dropout to every layer and try again. Does it help?
