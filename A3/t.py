import numpy as np
A=np.array([[1,2,3],[4,5,6]])
B=np.array([[7,8,9],[10,11,12]])
A=np.append(A,B,axis=0)
print(A)