# Unsupervised-Learning-and-Dimension-Reduction
Clustering Techniques

Link: https://github.com/dngu7/Unsupervised-Learning-and-Dimension-Reduction

# Setup
Ensure you have python3.6 and following packages installed.
tensorflow >=1.12.0
numpy >= 1.16.1
panda >= 0.3.1
sklearn >= 0.13.3
matplotlib >=3.0.2

Pull GitHub into local 

# Running the clustering algorithms and dimensionality reduction

argument 1: python
argument 2: unsupervisedlearning.py
argument 3: [data] = {"pima", "digits"}
argument 4: [kmeans] >=2
argument 5: [clustermode] ={"kmeans", "em"}
argument 5: [reductionmode] ={"None", "pca","ica,"random","NMF"}

Example 1: Expectation Maximisation clustering on digit dataset with ICA reduction
- python unsupervisedlearning.py pima 2 em ica


## Running the neural network with dimensionality reduction and clustering

# Data
Download the MNIST and CIFAR-100 datasets and place into their respective data folders
MNIST: http://yann.lecun.com/exdb/mnist/
CIFAR-100: https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz

To run the neural network, use the following arguments while in the directory

argument 1: python
argument 2: runner.py
argument 3: [algorithm] = {"nn"}
argument 4: [data] = {"mnist", "cifar"}
argument 4: [reduction] = {"pca", "ica", "rp","nmf","kmeans","em"}

Example 1: Neural Network on the MNIST database
- python runner.py nn mnist pca

Example 2: Support Vector Machine on the CIFAR database
- python runner.py nn cifar ica

