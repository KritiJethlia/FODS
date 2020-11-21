import numpy as np
def generate_param(X, n):
	new_X = np.zeros((X.shape[0],1))
	for degree in range(1,n+1):
		if degree==1:
			new_X = np.hstack((new_X, X))

		else:
			for i in range(0, degree+1):
				temp = (X[:,0]**(degree-i))*(X[:,1]**i)
				temp = np.reshape(temp, (X.shape[0], 1))
				new_X = np.hstack((new_X, temp))
		

	new_X = np.delete(new_X, 0, 1)
	return new_X

if __name__ == '__main__':
	X = np.array([[1,2], [3,4]])
	print(generate_param(X, 3))





