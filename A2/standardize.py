import numpy as np 
def standardize(X, mean = None, std = None):
	eps = 10e-6
	if mean is None or std is None:
		mean = np.mean(X, axis = 0)
		std = np.std(X, axis = 0)

	X = (X-mean)/(std + eps)
	return X, mean, std

