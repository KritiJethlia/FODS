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
	X = np.array(X)
	X = np.delete(X, 2, axis = 1)
	return X, np.array(y)

if __name__ == '__main__':
	X, y = get_data('insurance.txt')


