import numpy as np
from get_data import get_data
from split_data import split_data
from standardize import *
from generate_param import generate_param
import matplotlib.pyplot as plt

def stochastic_grad(train_X, valid_X, test_X, train_y, valid_y, test_y, lr):
    N = train_X.shape[1]  #number of parameters
    W = np.random.rand(N, 1)
    losses = []
    epochs=100
    for i in range(epochs):
        step_loss=[]
        for step in range(train_X.shape[0]):
            y_pred = np.dot(train_X[step:step+1, :],W)
            loss = y_pred - train_y[step:step+1, :]
            d_W = gradient(loss, train_X[step:step+1, :])
            W-= lr*d_W
            step_loss.append(rmse(loss))
            
        losses.append(np.mean(step_loss))
            
    return W, losses

def gradient(loss, train_X):
    d_theta = np.dot((train_X.T), loss)
    return d_theta



#Evaluates performance of model on given data
def evaluate(X, y, theta):
    y_pred = np.dot(X,theta)
    loss = y_pred-y
    loss = rmse(loss)
    return loss

def rmse(loss):
    rmse = np.sqrt(1/(loss.shape[0])*np.dot((loss.T), loss))
    return np.squeeze(rmse)

def main():
    X, y = get_data('insurance.txt')
    rmse_loss_train = [[], [], []]
    rmse_loss_test = [[], [], []]

    #split
    train_X, train_y, valid_X, valid_y, test_X, test_y = split_data(X, y)

    #standardize
    train_X, mean, std = standardize(train_X)
    valid_X, _, _ = standardize(valid_X, mean, std) 
    test_X, _, _ = standardize(test_X, mean, std) 

    learning_rates = [0.1, 0.01, 0.001]

    #calculating for a particular polynomial
    for j in learning_rates:
        for i in range(10,11):
            # print(train_X[0])
            train_X = add_bais(generate_param(train_X, i))
            valid_X = add_bais(generate_param(valid_X, i))
            test_X = add_bais(generate_param(test_X, i))
            # print(train_X[0].shape)
            # print(valid_X[0].shape)
            theta, losses = stochastic_grad(train_X, valid_X, test_X, train_y, valid_y, test_y, j)
            rmse_train = evaluate(train_X, train_y, theta)
            rmse_test = evaluate(test_X, test_y, theta)
            # rmse_loss_train[i].append(loss_train)
            # rmse_loss_test[i].append(loss_test)
            if i==10:
                plt.plot(losses)
                plt.xlabel('Epochs')
                plt.ylabel('RMSE loss')
                plt.ylim(0, 15000)
                plt.show()
main()