from shuffle_data import *
import numpy as np
def split_data(X, y):
	X, y = shuffle_data(X, y)
	n = len(y)
	train_X = X[:int(0.7*n)]
	train_y = y[:int(0.7*n)]
	valid_X = X[int(0.7*n):int(0.9*n)]
	valid_y =  y[int(0.7*n):int(0.9*n)]
	test_X = X[int(0.9*n):]
	test_y =  y[int(0.9*n):]

	return train_X, train_y, valid_X, valid_y, test_X, test_y
