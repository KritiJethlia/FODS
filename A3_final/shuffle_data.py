import random
def shuffle_data(X, y):
	indices = list(range(X.shape[0]))
	random.shuffle(indices)
	X = X[indices]
	y = y[indices]
	return X, y
