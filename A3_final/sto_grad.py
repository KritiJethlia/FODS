from get_data import get_data
from split_data import split_data
from standardize import *
import numpy as np
import matplotlib.pyplot as plt
from generate_param import generate_param
from mpl_toolkits import mplot3d
from sklearn.preprocessing import PolynomialFeatures

def main(lr):
    print(f'for learning rate {lr}')
    X, y = get_data('insurance.txt')
    y=y*(1/10000)

    #Split Data
    train_X, train_y, valid_X, valid_y, test_X, test_y = split_data(X, y)
    m = train_X.shape[0]
    n = train_X.shape[1]
    m_test= test_X.shape[0]
    m_valid= valid_X.shape[0]

    #Standardization
    # train_X, mean, std = standardize(train_X)
    # test_X, _, _ = standardize(test_X, mean, std) 
    # valid_X, _, _ = standardize(valid_X, mean, std)
    #Mean and standard deviation is calculated for train data and this mean and std are used for standardising the test data 
    

    #for polynomials of size 1 to 10
    for j in range(1,11) :
        print(f'\n-----------for polynomial of degree {j}------------ ')

        train_X = generate_param(train_X, j)
        test_X = generate_param(test_X, j)
        valid_X = generate_param(valid_X, j)
        train_X, mean, std = standardize(train_X)
        test_X, _, _ = standardize(test_X, mean, std) 
        valid_X, _, _ = standardize(valid_X, mean, std)
        train_X_param = add_bias(train_X)
        test_X_param = add_bias(test_X)
        valid_X_param = add_bias(valid_X)

        # train_X_param = add_bias(generate_param(train_X, j))
        # test_X_param = add_bias(generate_param(test_X, j))
        # valid_X_param = add_bias(generate_param(valid_X, j))

        theta = np.random.rand(train_X_param.shape[1], 1)
        
        #for normal stochastic gradient descent
        theta, losses = train_model(train_X_param, train_y, theta, lr)
        print('---------Normal Stochastic Gradient Descent---------')
        rmse_train = evaluate(train_X_param, train_y, theta)
        print(f'RMSE train error : {rmse_train}')
        rmse_valid = evaluate(np.append(test_X_param,valid_X_param,axis=0), np.append(test_y,valid_y,axis=0), theta)
        print(f'RMSE test error : {rmse_valid}')
        

        #for stochastic gradient descent with L1(Lasso)
        theta = np.random.rand(train_X_param.shape[1], 1)
        theta, losses = train_model_l1(train_X_param, train_y, theta, valid_X_param, valid_y, lr)
        print('---------SGD with Lasso reg ---------')
        rmse_train = evaluate(train_X_param, train_y, theta)
        print(f'RMSE train error : {rmse_train}')
        rmse_valid = evaluate(valid_X_param, valid_y, theta)
        print(f'RMSE validate error : {rmse_valid}')
        rmse_test = evaluate(test_X_param, test_y, theta)
        print(f'RMSE test error : {rmse_test}')


        #for stochastic gradient descent with L2(Ridge)
        theta = np.random.rand(train_X_param.shape[1], 1)
        theta, losses = train_model_l2(train_X_param, train_y, theta, valid_X_param, valid_y, lr)
        print('---------SGD with Ridge reg ---------')
        rmse_train = evaluate(train_X_param, train_y, theta)
        print(f'RMSE train error : {rmse_train}')
        rmse_valid = evaluate(valid_X_param, valid_y, theta)
        print(f'RMSE validate error : {rmse_valid}')
        rmse_test = evaluate(test_X_param, test_y, theta)
        print(f'RMSE test error : {rmse_test}')


        #graph
        # X_axis = train_X[:,0]
        # Y_axis = train_X[:,1]
        # X_axis, Y_axis = np.meshgrid(X_axis,Y_axis)
        # Z_axis = np.dot(add_bias(generate_param(train_X,j)),theta)
        # fig = plt.figure()
        # ax = plt.axes(projection="3d")
        # ax.set_xlabel('Age')
        # ax.set_ylabel('BMI')
        # ax.set_zlabel('Charges')
        # ax.plot_surface(X_axis, Y_axis, Z_axis, rstride=1, cstride=1,cmap='winter', edgecolor='none')
        # ax.set_title(f'surface plot for polynomial of degree {j} ')
        # # plt.show()
        # plt.savefig("normal_"+ str(i) +'.png',dpi=90)
        
    return rmse_train, rmse_valid, losses

def add_bias(X):
    temp = np.ones((X.shape[0],X.shape[1]+1))
    temp[:,1:] = X
    return temp

def train_model(train_X, train_y, theta, lr):
    losses = []
    #Dividing each episode in steps
    for episode in range(1000):
        step_loss = []
        for step in range(train_X.shape[0]):
            y_pred = np.dot(train_X[step:step+1, :],theta)
            loss = y_pred-train_y[step:step+1, :]
            d_theta = gradient(loss, train_X[step:step+1, :])
            theta-=lr*d_theta
            step_loss.append(rmse(loss))
            
        # losses.append(np.mean(step_loss))
        # if not episode%50:
        #     print(f'Root mean square error at episode {episode} is {losses[-1]}')
            
    return theta, losses

def train_model_l1(train_X, train_y, theta, valid_X, valid_y, lr):
    losses = []
    lambd = np.linspace(0.1,1,5)
    #Dividing each episode in steps
    sig = lambda x: (x/abs(x))
    theta_arr=[]
    rmse_valid=[]
    for l1 in lambd :
        theta = np.random.rand(train_X.shape[1], 1)
        for episode in range(1000):
            # step_loss = []
            for step in range(train_X.shape[0]):
                y_pred = np.dot(train_X[step:step+1, :],theta)
                loss = y_pred-train_y[step:step+1, :]
                sign_theta = np.array([sig(t) for t in theta])
                d_theta = gradient(loss, train_X[step:step+1, :])+ l1*sign_theta
                theta-=lr*d_theta
                # step_loss.append(rmse(loss))
        theta_arr.append(theta)   
        rmse_valid.append(evaluate(valid_X, valid_y, theta))
        # losses+=[np.mean(step_loss)]
        # print(f'Root mean square error at lambda {l1} is {losses[-1]}')
            # if not episode%50:
            #     print(f'Root mean square error at episode {episode} is {losses[-1]}')
    index = rmse_valid.index(min(rmse_valid))   
    print(f'Optimal Lambda : {lambd[index]}')    
    return theta_arr[index], losses

def train_model_l2(train_X, train_y, theta, valid_X, valid_y, lr):
    losses = []
    lambd = np.linspace(0.1,1,5)
    theta_arr=[]
    rmse_valid=[]
    for l2 in lambd:
        theta = np.random.rand(train_X.shape[1], 1)
        for episode in range(1000):
            # step_loss = []
            for step in range(train_X.shape[0]):
                y_pred = np.dot(train_X[step:step+1, :],theta)
                loss = y_pred-train_y[step:step+1, :]
                d_theta = gradient(loss, train_X[step:step+1, :])+ l2*theta
                theta-=lr*d_theta
                # step_loss.append(rmse(loss))
        theta_arr+=[theta]   
        rmse_valid+=[evaluate(valid_X, valid_y, theta)]    
        # losses.append(np.mean(step_loss))
        # print(f'Root mean square error at lambda {l2} is {losses[-1]}')
        # if not episode%50:
        #     print(f'Root mean square error at episode {episode} is {losses[-1]}')
    index = rmse_valid.index(min(rmse_valid))       
    print(f'Optimal Lambda : {lambd[index]}')
    return theta_arr[index], losses

#Evaluates performance of model on given data
def evaluate(X, y, theta):
    y_pred = np.dot(X,theta)
    loss = y_pred-y
    loss = rmse(loss)
    return loss

#Calculates grdient of loss with respect to all parameters
def gradient(loss, train_X):
    d_theta = np.dot((train_X.T), loss)
    return d_theta

#Calculates rmse loss 
def rmse(loss):
    rmse = np.sqrt(1/(loss.shape[0])*np.dot((loss.T), loss))
    return np.squeeze(rmse)


if __name__ == '__main__':
    learning_rates = [0.00001]
    for i in range(len(learning_rates)):
        main(learning_rates[i])
    