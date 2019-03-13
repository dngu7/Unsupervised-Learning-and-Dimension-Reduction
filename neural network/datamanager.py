from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import gzip
import collections
import numpy
from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import random_seed
from tensorflow.python.platform import gfile
from tensorflow.python.util.deprecation import deprecated
from sklearn.decomposition import PCA, FastICA, NMF

DEFAULT_SOURCE_URL = 'https://storage.googleapis.com/cvdf-datasets/mnist/'

Datasets = collections.namedtuple('Datasets', ['train', 'validation', 'test'])


def _read32(bytestream):
	dt = numpy.dtype(numpy.uint32).newbyteorder('>')
	return numpy.frombuffer(bytestream.read(4), dtype=dt)[0]


def dense_to_one_hot(labels_dense, num_classes):
	"""Convert class labels from scalars to one-hot vectors."""
	num_labels = labels_dense.shape[0]
	index_offset = numpy.arange(num_labels) * num_classes
	labels_one_hot = numpy.zeros((num_labels, num_classes))
	labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
	return labels_one_hot

def unpickle(file):
	import pickle
	with open(file, 'rb') as fo:
		dict = pickle.load(fo, encoding='bytes')
	return dict


class DataSet(object):
	def __init__(self, images,labels,one_hot=False,dtype=dtypes.float32,reshape=True,seed=None,reduct=None):

		seed1, seed2 = random_seed.get_seed(seed)

		# If op level seed is not set, use whatever graph level seed is returned
		numpy.random.seed(seed1 if seed is None else seed2)
		dtype = dtypes.as_dtype(dtype).base_dtype
		if dtype not in (dtypes.uint8, dtypes.float32):
			raise TypeError('Invalid image dtype %r, expected uint8 or float32' % dtype)


		self._num_examples = images.shape[0]

		# Convert shape from [num examples, rows, columns, depth]
		# to [num examples, rows*columns] (assuming depth == 1)
		if reshape:
			assert images.shape[3] == 1
			images = images.reshape(images.shape[0],images.shape[1] * images.shape[2])
		if dtype == dtypes.float32:
			# Convert from [0, 255] -> [0.0, 1.0].
			images = images.astype(numpy.float32)
			images = numpy.multiply(images, 1.0 / 255.0)

		if reduct == "pca":
			pca = PCA(svd_solver='full')
			images = pca.fit_transform(images)
		elif reduct == 'ica':
			ica = FastICA(fun='cube', random_state=0,tol=0.1,max_iter=1000)
			images = ica.fit_transform(images)
		elif reduct == 'rp':
			random_ = random_projection.GaussianRandomProjection()
			images = random_.fit_transform(images)
		elif reduct == 'nmf':
			nmf = NMF()
			images = nmf.fit_transform(images)
		else:
			pca = PCA(svd_solver='full')
			images = pca.fit_transform(images)			
		

		self._images = images
		self._labels = labels
		self._epochs_completed = 0
		self._index_in_epoch = 0



	@property
	def images(self):
		return self._images

	@property
	def labels(self):
		return self._labels

	@property
	def num_examples(self):
		return self._num_examples

	@property
	def epochs_completed(self):
		return self._epochs_completed

	def next_batch(self, batch_size, shuffle=True):
		"""Return the next `batch_size` examples from this data set."""


		start = self._index_in_epoch
		# Shuffle for the first epoch
		if self._epochs_completed == 0 and start == 0 and shuffle:
			perm0 = numpy.arange(self._num_examples)
			numpy.random.shuffle(perm0)
			self._images = self.images[perm0]
			self._labels = self.labels[perm0]

		# Go to the next epoch
		if start + batch_size > self._num_examples:
			# Finished epoch
			self._epochs_completed += 1
			# Get the rest examples in this epoch
			rest_num_examples = self._num_examples - start
			images_rest_part = self._images[start:self._num_examples]
			labels_rest_part = self._labels[start:self._num_examples]
			# Shuffle the data
			if shuffle:
				perm = numpy.arange(self._num_examples)
				numpy.random.shuffle(perm)
				self._images = self.images[perm]
				self._labels = self.labels[perm]
			# Start next epoch
			start = 0
			self._index_in_epoch = batch_size - rest_num_examples
			end = self._index_in_epoch
			images_new_part = self._images[start:end]
			labels_new_part = self._labels[start:end]
			return numpy.concatenate((images_rest_part, images_new_part), axis=0), numpy.concatenate((labels_rest_part, labels_new_part), axis=0)
		else:
			self._index_in_epoch += batch_size
			end = self._index_in_epoch
			return self._images[start:end], self._labels[start:end]



def cifar_read_data_sets(train_dir,reduct,one_hot=False,dtype=dtypes.float32,reshape=True,validation_size=5000,seed=None, n_classes=20):

	cifar_train_data = train_dir + 'train'

	unpk_train_data = unpickle(cifar_train_data)
	train_images = numpy.array(unpk_train_data[b'data'])
	train_labels = numpy.array(unpk_train_data[b'coarse_labels'])

	cifar_test_data = train_dir + 'test'

	unpk_test_data = unpickle(cifar_test_data)
	test_images = numpy.array(unpk_test_data[b'data'])
	test_labels = numpy.array(unpk_test_data[b'coarse_labels'])
	if one_hot: 
		train_labels = dense_to_one_hot(train_labels,n_classes)
		test_labels = dense_to_one_hot(test_labels,n_classes)

	validation_images = train_images[:validation_size]
	validation_labels = train_labels[:validation_size]
	train_images = train_images[validation_size:]
	train_labels = train_labels[validation_size:]

	options = dict(dtype=dtype, reshape=reshape, seed=seed,reduct=reduct)
	train = DataSet(train_images, train_labels, **options)
	validation = DataSet(validation_images, validation_labels, **options)  
	test = DataSet(test_images, test_labels, **options)  
	return Datasets(train=train, validation=validation, test=test)

def maybe_download(filename, work_directory, source_url):
	"""Download the data from source url, unless it's already here.

	Args:
			filename: string, name of the file in the directory.
			work_directory: string, path to working directory.
			source_url: url to download from if file doesn't exist.

	Returns:
			Path to resulting file.
	"""
	if not gfile.Exists(work_directory):
		gfile.MakeDirs(work_directory)
	filepath = os.path.join(work_directory, filename)
	if not gfile.Exists(filepath):
		temp_file_name, _ = urlretrieve_with_retry(source_url)
		gfile.Copy(temp_file_name, filepath)
		with gfile.GFile(filepath) as f:
			size = f.size()
		print('Successfully downloaded', filename, size, 'bytes.')
	return filepath

def extract_images(f):
	"""Extract the images into a 4D uint8 numpy array [index, y, x, depth].

	Args:
		f: A file object that can be passed into a gzip reader.

	Returns:
		data: A 4D uint8 numpy array [index, y, x, depth].

	Raises:
		ValueError: If the bytestream does not start with 2051.

	"""
	print('Extracting', f.name)
	with gzip.GzipFile(fileobj=f) as bytestream:
		magic = _read32(bytestream)
		if magic != 2051:
			raise ValueError('Invalid magic number %d in MNIST image file: %s' %
											 (magic, f.name))
		num_images = _read32(bytestream)
		rows = _read32(bytestream)
		cols = _read32(bytestream)
		buf = bytestream.read(rows * cols * num_images)
		data = numpy.frombuffer(buf, dtype=numpy.uint8)
		data = data.reshape(num_images, rows, cols, 1)
		return data

def extract_labels(f, one_hot=False, num_classes=10):
	"""Extract the labels into a 1D uint8 numpy array [index].

	Args:
		f: A file object that can be passed into a gzip reader.
		one_hot: Does one hot encoding for the result.
		num_classes: Number of classes for the one hot encoding.

	Returns:
		labels: a 1D uint8 numpy array.

	Raises:
		ValueError: If the bystream doesn't start with 2049.
	"""
	print('Extracting', f.name)
	with gzip.GzipFile(fileobj=f) as bytestream:
		magic = _read32(bytestream)
		if magic != 2049:
			raise ValueError('Invalid magic number %d in MNIST label file: %s' %
											 (magic, f.name))
		num_items = _read32(bytestream)
		buf = bytestream.read(num_items)
		labels = numpy.frombuffer(buf, dtype=numpy.uint8)
		if one_hot:
			return dense_to_one_hot(labels, num_classes)
		return labels

def mnist_read_data_sets(train_dir,
									 reduct,
									 fake_data=False,
									 one_hot=False,
									 dtype=dtypes.float32,
									 reshape=True,
									 validation_size=5000,
									 seed=None,
									 source_url=DEFAULT_SOURCE_URL):

	TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
	TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
	TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
	TEST_LABELS = 't10k-labels-idx1-ubyte.gz'

	local_file = maybe_download(TRAIN_IMAGES, train_dir,
																	 source_url + TRAIN_IMAGES)
	with gfile.Open(local_file, 'rb') as f:
		train_images = extract_images(f)

	local_file = maybe_download(TRAIN_LABELS, train_dir,
																	 source_url + TRAIN_LABELS)
	with gfile.Open(local_file, 'rb') as f:
		train_labels = extract_labels(f, one_hot=one_hot)

	local_file = maybe_download(TEST_IMAGES, train_dir,
																	 source_url + TEST_IMAGES)
	with gfile.Open(local_file, 'rb') as f:
		test_images = extract_images(f)

	local_file = maybe_download(TEST_LABELS, train_dir,
																	 source_url + TEST_LABELS)
	with gfile.Open(local_file, 'rb') as f:
		test_labels = extract_labels(f, one_hot=one_hot)

	if not 0 <= validation_size <= len(train_images):
		raise ValueError('Validation size should be between 0 and {}. Received: {}.'
										 .format(len(train_images), validation_size))

	validation_images = train_images[:validation_size]
	validation_labels = train_labels[:validation_size]
	train_images = train_images[validation_size:]
	train_labels = train_labels[validation_size:]

	options = dict(dtype=dtype, reshape=reshape, seed=seed)

	train = DataSet(train_images, train_labels, **options)
	validation = DataSet(validation_images, validation_labels, **options)
	test = DataSet(test_images, test_labels, **options)

	return Datasets(train=train, validation=validation, test=test)








