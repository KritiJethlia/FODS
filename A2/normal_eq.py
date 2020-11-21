import numpy as np
from get_data import get_data
from split_data import *
from standardize import *

def normal_eq_calc(X,y,test_X,test_y):
    ''' 
    Function to calculate the parameters using normal equation method.
    Takes test and train values and returns respective rmse error.
    '''
    X_matrix=np.ones((X.shape[0],X.shape[1]+1)) #X_matrix is matrix with all ones
    X_matrix[:, 1:] = X # now X_matrix has first column 1 and other columns same as X
    X_transpose = X_matrix.T
    theta = np.linalg.inv(X_transpose.dot(X_matrix)).dot(X_transpose).dot(y) # theta =inverse(XT . X) . XT
    rmse_error_train = rmse(theta, X, y)
    rmse_error_test = rmse(theta, test_X, test_y)
    return rmse_error_train , rmse_error_test

def rmse(coeff, data_X, data_y):
    '''
    Function to calculate rmse errors for given data.
    Takes coeff and data as input and returns rmse as output.
    '''
    variable_matrix = np.ones((data_X.shape[0],data_X.shape[1]+1))
    variable_matrix[:, 1:] = data_X
    predicted_matrix = variable_matrix.dot(coeff) #predicted value
    error_matrix = data_y - predicted_matrix #difference
    error = error_matrix.T.dot(error_matrix) # Et . E
    return np.sqrt(error[0][0]/data_X.shape[0])

def main():
    rmse_test = []
    rmse_train = []
    X ,y = get_data('insurance.txt')
    for i in range(0,20) :
        train_X, train_y, test_X, test_y = split_data(X,y) #70:30 split
        val_train , val_test = normal_eq_calc(train_X ,train_y ,test_X ,test_y)
        rmse_train +=  [val_train]
        rmse_test += [val_test]

    rmse_train = np.array(rmse_train)
    rmse_test = np.array(rmse_test)
    print("Mean of train data : ", np.mean(rmse_train))
    print("Variance of train data :  ", np.var(rmse_train))
    print("Minimum of train data : ", np.amin(rmse_train),"\n")
    print("Mean of test data : ", np.mean(rmse_test))
    print("Variance of test data : ", np.var(rmse_test))
    print("Minimum of test data : ", np.amin(rmse_test))

main()