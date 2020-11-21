from shuffle_data import *
import numpy as np
def split_data(X, y):
	X, y = shuffle_data(X, y)
	n = len(y)
	train_X = X[:int(0.7*n)]
	train_y = y[:int(0.7*n)]
	test_X = X[int(0.7*n):]
	test_y =  y[int(0.7*n):]

	return train_X, train_y, test_X, test_y
