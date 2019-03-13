from Layers.nn import *
import tensorflow as tf
import numpy as np
import os.path
import sys
from sklearn.metrics import accuracy_score, mean_squared_error
from datetime import datetime
import datamanager


from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split, learning_curve, ShuffleSplit
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix  

import numpy as np  
import matplotlib.pyplot as plt  
import pandas as pd  




mnist_dir = './data/mnist/'
cifar_dir = './data/cifar/'

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Accuracy")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training accuracy")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross Validation accuracy")

    plt.legend(loc="best")
    return plt




def neuralnet(current_dataset, data):
    learning_rate = 0.001
    batch_size = 256
    n_training_epochs = 100

    if data == "mnist":
        n_features = 784 #784
        n_classes = 10
        dim_1 = 28 #28
        dim_2 = 28
        dim_3 = 1
        

    elif data == "cifar":
        n_features = 3072
        n_classes = 20
        dim_1 = 32 
        dim_2 = 32
        dim_3 = 1
        

    reshape_size = n_features * n_classes

    # Input (X) and Target (Y) placeholders, they will be fed with a batch of
    # input and target values respectively, from the training and test sets
    X = input_placeholder(n_features)
    Y = target_placeholder(n_classes)

    #convolutional neural network
    logits_op, preds_op, loss_op = \
        convnet(tf.reshape(X, [-1, dim_1, dim_2, dim_3]), Y, convlayer_sizes=[n_classes, n_classes],
                     filter_shape=[3, 3], outputsize=n_classes, padding="same", reshape_size=reshape_size)
    tf.summary.histogram('pre_activations', logits_op)

   # The training op performs a step of stochastic gradient descent on a minibatch
    optimizer = tf.train.AdamOptimizer  # ADAM - widely used optimiser (ref: http://arxiv.org/abs/1412.6980)
    train_op = optimizer(learning_rate).minimize(loss_op)

    # Prediction and accuracy ops
    accuracy_op = get_accuracy_op(preds_op, Y)

    # TensorBoard for visualisation
    # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
    summaries_op = tf.summary.merge_all()

    # Separate accuracy summary so we can use train and test sets
    accuracy_placeholder = tf.placeholder(shape=[], dtype=tf.float32)
    accuracy_summary_op = tf.summary.scalar("accuracy", accuracy_placeholder)

    # When run, the init_op initialises any TensorFlow variables
    # hint: weights and biases in our case
    init_op = tf.global_variables_initializer()

    # Get started
    sess = tf.Session()
    sess.run(init_op)

    # Initialise TensorBoard Summary writers
    dtstr = "{:%b_%d_%H-%M-%S}".format(datetime.now())
    train_writer = tf.summary.FileWriter('./summaries/' + dtstr + '/train', sess.graph)
    test_writer = tf.summary.FileWriter('./summaries/' + dtstr + '/test')

    # Train
    print('Starting Training...')
    train_accuracy, test_accuracy = nn_train(sess, current_dataset, n_training_epochs, batch_size,
                                          summaries_op, accuracy_summary_op, train_writer, test_writer,
                                          X, Y, train_op, loss_op, accuracy_op, accuracy_placeholder)
    print('Training Complete\n')
    print("train_accuracy: {train_accuracy}, test_accuracy: {test_accuracy}".format(**locals()))

    # Clean up
    sess.close()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("method", choices=["nn"])
    parser.add_argument("data", choices=["mnist", "cifar"])
    parser.add_argument("reduction", choices=["pca", "ica", "rp","nmf","kmeans","em"])

    args = parser.parse_args()

    group1 = ["tree", "nn"]

    if args.method in group1:
        if args.data == "mnist":
            current_dataset = datamanager.mnist_read_data_sets(train_dir=mnist_dir, reduct=args.reduction, one_hot=True, reshape=True)
        elif args.data == "cifar":
            current_dataset = datamanager.cifar_read_data_sets(train_dir=cifar_dir,reduct=args.reduction, one_hot=True, reshape=False, n_classes=20)  
    else:
        if args.data == "mnist":
            current_dataset = datamanager.mnist_read_data_sets(train_dir=mnist_dir,reduct=args.reduction, one_hot=False, reshape=True)
        elif args.data == "cifar":
            current_dataset = datamanager.cifar_read_data_sets(train_dir=cifar_dir,reduct=args.reduction, one_hot=False, reshape=False, n_classes=20)  
                

    neuralnet(current_dataset, args.data)






