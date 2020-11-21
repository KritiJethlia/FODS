import numpy as np
def get_data(path):
	X = []
	y = []
	with open(path, 'r') as file:
		file.readline()
		for line in file:
			data = line.split(',')
			data = list(map(float, data))
			X.append(data[:-1])
			y.append(data[-1])

	y = np.reshape(y, (len(y), 1))
	return np.array(X), np.array(y)
