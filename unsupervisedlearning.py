# Dependencies

%matplotlib inline
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import scale

from sklearn.metrics import accuracy_score
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.datasets import load_digits
from scipy.stats import mode

from sklearn.decomposition import PCA, FastICA, NMF
from sklearn import random_projection


import seaborn as sns
import matplotlib.pyplot as plt
import sys

datatype = sys.argv[1]
k_value_max = sys.argv[2]
clustermode = sys.argv[3]

if len(sys.argv)>=5 and sys.argv[4]:
	featuremode = sys.argv[4]
else:
	featuremode = None

if datatype not in ['pima','digits'] or \
	clustermode not in ['em','kmeans'] or \
	featuremode not in [None,'PCA','ICA','random','NMF']:
	print("Usage: python3 clustering.py datatype loopcount clustermode featureselection")
	print("k_value_max: <int>")
	print("datatype: pima, digits")
	print("clustermode: kmeans, em")
	print("featureselection: None, PCA, LCA, NMF, random")
	sys.exit()


accuracy_list = []
x_axis = []
for i in range(2,int(k_value_max)):

	####################################################
	#### DATA PREPARATION

	if datatype == 'pima':
		k_value = i
		data_location = './pima-indians-diabetes.data'
		header = ['pregnancies', 'glucose', 'bloodpressure', 'skinthickness', 
				'insulin', 'bmi', 'diabetepedigree', 'age', 'outcome']
		
		df = pd.read_csv(data_location, names=header)

		X = np.array(df.drop(['outcome'], 1).astype(float))
		y = np.array(df['outcome'])


	elif datatype == 'digits':
		k_value = i #10
		digits = load_digits()
		X = digits.data
		y = digits.target


	####################################################
	#### FEATURE TRANSFORMATION

	if featuremode == 'PCA':
		pca = PCA(n_components=k_value,svd_solver='full')
		X = pca.fit_transform(X)

	elif featuremode == 'ICA':
		ica = FastICA(n_components=k_value, fun='cube', random_state=0,tol=0.1,max_iter=1000)
		X = ica.fit_transform(X)
		#Notes: Improves accuracy from 0.67 to 0.72 for pima indians
elif featuremode == 'random':
		random_ = random_projection.GaussianRandomProjection(n_components=k_value)
		X = random_.fit_transform(X)
		#accuracy is 0.65
	elif featuremode == 'NMF':
		nmf = NMF(n_components=k_value)
		X = nmf.fit_transform(X)
		#Notes: reduces accuracy from 0.67 to 0.65 for pima indians

	
	####################################################
	#### BUILD CLUSTERS 

	if clustermode == 'kmeans':
		kmeans = KMeans(init='k-means++', n_clusters=k_value, random_state=0)
		clusters = kmeans.fit_predict(X)



	elif clustermode == 'em':
		gmm = GaussianMixture(n_components=k_value,
									covariance_type='full',
									max_iter=600,
									n_init=1)

		clusters = gmm.fit_predict(X)
		#probs = gmm.predict_proba(X)
		#print(probs[:100].round(3))

	####################################################
	####  

	predict_y = np.zeros_like(clusters)
	for i in range(k_value):
	    mask = (clusters == i)
	    predict_y[mask] = mode(y[mask])[0]

	x_axis.append(i)
	accuracy_list.append(accuracy_score(y, predict_y))

plt.scatter(x_axis, accuracy_list, s=10, alpha=0.5)
plt.title('Cluster Size Impact on Accuracy')
plt.xlabel('Cluster Size')
plt.ylabel('Accuracy')
plt.show()
	#0.667

