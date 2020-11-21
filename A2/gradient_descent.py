from get_data import get_data
from split_data import split_data
from standardize import *
import numpy as np
import matplotlib.pyplot as plt

def main(lr = 0.01, flag = False):
    X, y = get_data('insurance.txt')

    #Split Data
    train_X, train_y, test_X, test_y = split_data(X, y)

    m = train_X.shape[0]
    n = train_X.shape[1]
    m_test= test_X.shape[0]

    #Standardization
    #Mean and standard deviation is calculated for train data and this mean and std are used for standardising the test data 
    train_X, mean, std = standardize(train_X)
    test_X, _, _ = standardize(test_X, mean, std) 

    #Adding a column of ones for bias term 
    temp = np.ones((m, n+1))
    temp[:, 1:] = train_X
    train_X = temp

    temp = np.ones((m_test, n+1))
    temp[:, 1:] = test_X
    test_X = temp


    print(train_X.shape)
    print(train_y.shape)
    print(test_X.shape)
    print(test_y.shape)

    m = train_X.shape[0]
    n = train_X.shape[1]
    m_test= test_X.shape[0]
    
    #Visualizing the data for better insight
    if flag:
        visualize(train_X, train_y)

    theta = np.random.rand(n, 1)
    theta, losses = train_model(train_X, train_y, theta, lr)
    rmse_train = evaluate(train_X, train_y, theta)
    rmse_test = evaluate(test_X, test_y, theta)
    print(theta)
    return rmse_train, rmse_test, losses

def train_model(train_X, train_y, theta, lr):
    losses = []
    for i in range(10000):
        y_pred = np.dot(train_X,theta)
        loss = y_pred-train_y
        d_theta = gradient(loss, train_X)
        theta-=lr*d_theta
        losses.append(rmse(loss))
        if not i%50:
            print('Root mean square error at episode {i} is {losses[-1]}')
            
    return theta, losses

#Evaluates performance of model on given data
def evaluate(X, y, theta):
    y_pred = np.dot(X,theta)
    loss = y_pred-y
    loss = rmse(loss)
    return loss

#Calculates grdient of loss with respect to all parameters
def gradient(loss, train_X):
    d_theta = 1/(train_X.shape[0])*np.dot((train_X.T), loss)
    return d_theta

#Calculates rmse loss 
def rmse(loss):
    rmse = np.sqrt(1/(loss.shape[0])*np.dot((loss.T), loss))
    return np.squeeze(rmse)

#Plotting y wrt other parameters to gain insight on data
def visualize(train_X, train_y):
    plt.scatter(x = train_X[:,1], y = train_y)
    plt.xlabel('Age')
    plt.ylabel('Insurance')
    plt.show()
    plt.scatter(x = train_X[:,2], y = train_y)
    plt.xlabel('BMI')
    plt.ylabel('Insurance')
    plt.show()
    plt.scatter(x = train_X[:,3], y = train_y)
    plt.xlabel('Number of Children')
    plt.ylabel('Insurance')
    plt.show()
    plt.plot(sorted(train_y))
    plt.show()

if __name__ == '__main__':
    rmse_loss_train = [[], [], []]
    rmse_loss_test = [[], [], []]
    learning_rates = [0.1, 0.01, 0.001]
    for i in range(len(learning_rates)):
        for j in range(20):
            flag = False
            if i==0 and j==0:
                flag = True
            loss_train, loss_test, losses = main(learning_rates[i], flag)
            rmse_loss_train[i].append(loss_train)
            rmse_loss_test[i].append(loss_test)
            if j==0:
                plt.plot(losses)
                plt.xlabel('Epochs')
                plt.ylabel('RMSE loss')
                plt.show()	

    rmse_loss_train = np.array(rmse_loss_train)
    rmse_loss_test = np.array(rmse_loss_test)

    for i in range(3):
        print(f'Mean train loss at learning rate {learning_rates[i]} is', str(np.mean(rmse_loss_train[i])))
        print(f'Mean test loss at learning rate {learning_rates[i]} is', str(np.mean(rmse_loss_test[i])))

        print(f'Variance train loss at learning rate {learning_rates[i]} is' ,str(np.var(rmse_loss_train[i])))
        print(f'Variance test loss at learning rate {learning_rates[i]} is', str(np.var(rmse_loss_test[i])))

        print(f'Minimum train loss at learning rate {learning_rates[i]} is' ,str(np.min(rmse_loss_train[i])))
        print(f'Minimum test loss at learning rate {learning_rates[i]} is', str(np.min(rmse_loss_test[i])))