from get_data import get_data
from split_data import split_data
from standardize import *
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from math import sqrt

def evaluate(X, y, theta):
    y_pred = np.dot(X,theta)
    loss = y_pred-y
    loss = rmse(loss)
    return loss

def rmse(loss):
    rmse = np.sqrt(1/(loss.shape[0])*np.dot((loss.T), loss))
    return np.squeeze(rmse)

def gradient(loss, train_X):
    d_theta = 1/(train_X.shape[0])*np.dot((train_X.T), loss)
    return d_theta

def train_model(train_X, train_y, theta, lr):
    losses = []
    for i in range(10000):
        y_pred = np.dot(train_X,theta)
        loss = y_pred-train_y
        d_theta = gradient(loss, train_X)
        theta-=lr*d_theta
        losses.append(rmse(loss))
        #if not i%50:
            #print(f'Root mean square error at episode {i} is {losses[-1]}')
    return theta, losses

def score(weights,x,y):
        ss_tot = sum(np.square(np.mean(y) - y))
        ss_res = sum(np.square((x @ weights) - y))
        test_err = (0.5/len(x)) * ss_res
        print("avg. test err =", test_err )
        rmse = sqrt(ss_res/len(x))
        r2 = (1-(ss_res/ss_tot))
        return r2*100, rmse
        
def main(lr=0.00001):
    X,y=get_data('insurance.txt')
    l1_vals=[]
    l2_vals=[]
    for deg in range(1,11):
        poly_features=PolynomialFeatures(degree=deg)
        X_poly=poly_features.fit_transform(X)
        y=y*(1/10000)
        train_X, train_y, valid_X, valid_y, test_X, test_y=split_data(X_poly,y)
        train_X, mean, std = standardize(train_X)
        test_X, _, _ = standardize(test_X, mean, std)
        valid_X, _, _= standardize(valid_X, mean, std)
        train_X[:, 0] =1
        test_X[:,0]=1
        valid_X[:,0]=1
        m = train_X.shape[0]
        n = train_X.shape[1]
        m_test= test_X.shape[0]
        theta = np.random.rand(n, 1)
        theta, losses = train_model(train_X, train_y, theta, lr)
        rmse_train = evaluate(train_X, train_y, theta)
        rmse_valid = evaluate(valid_X, valid_y, theta)
        print("---------degree = ",deg)
        print("-------Weights and errors from gradient descent-------")
        print(theta)
        print(rmse_train, rmse_valid) #, losses
        print("-----------L1 regression----------")
        theta_L1,l1=L1_reg(lr, train_X, train_y, valid_X, valid_y,test_X, test_y)
        print("-----------L2 regression----------")
        theta_L2,l2=L2_reg(lr, train_X, train_y, valid_X, valid_y,test_X, test_y)
        l1_vals.append(l1)
        l2_vals.append(l2)
    return l1_vals,l2_vals
def L1_reg(lr, train_X, train_y, valid_X, valid_y,test_X,test_y):
    final_theta=np.array([], dtype=np.float64)
    validation_errors=[]
    l1_final=0
    dummy=1e10
    m = train_X.shape[0]
    n = train_X.shape[1]
    fun = lambda x: (x / abs(x))
    L1_values=np.linspace(0.1,0.9,5)
    for l1 in L1_values:
        theta = np.random.rand(n,1)
        previous_error = 1e10
        for i in range(10000):
            loss=np.dot(train_X,theta)-train_y
            err = 0.5 * (np.dot(loss.T, loss) + l1*sum([w*w for w in theta]))
            abs_w = np.array([fun(w) for w in theta])
            theta -= lr * ((np.dot((train_X.T),loss))) + lr*(0.5*l1*abs_w)
            if abs(previous_error-err) < 5e-3:
                    break
            previous_error = err
            loss_validation=np.dot(valid_X ,theta) - valid_y
            loss_training=np.dot(train_X , theta) - train_y
            VLE = (0.5/(valid_X.shape[0])) * sum(np.dot(loss_validation.T, loss_validation))
            ERR = (0.5/(train_X.shape[0])) * sum(np.dot(loss_training.T,loss_training))
        validation_errors.append(abs(ERR-VLE))
        if abs(ERR-VLE) < dummy:
            final_theta = theta
            l1_final = l1
            dummy = abs(ERR-VLE)
    print( "Optimal value of lambda for L1 ",l1_final)
    print(final_theta)
    print(score(final_theta,test_X,test_y))
    plt.plot(L1_values, validation_errors)
    plt.show()
    return final_theta,l1_final
def L2_reg(lr, train_X, train_y, valid_X, valid_y,test_X,test_y):
        final_theta=np.array([], dtype=np.float64)
        validation_errors = []
        l2_final = 0
        dummy = 1e10
        m = train_X.shape[0]
        n = train_X.shape[1]
        L2_vals = np.linspace(0.1, 0.9, 5)
        for l2 in L2_vals:
            previous_error = 1e10
            theta = np.random.rand(n,1)
            for i in range(10000):
                loss=np.dot(train_X,theta)-train_y
                err = 0.5 * (np.dot(loss.T, loss) + l2*sum([w*w for w in theta]))
                #if i % 50 == 0:
                    #print("epoch =", i , "| err_diff =", prev_err-err)
                    #print("error = ", err, "||", theta)
                theta -= lr * ((np.dot((train_X.T),loss))) + lr*(l2*theta)
                if abs(previous_error-err) < 5e-3:
                    break
                previous_error = err
            loss_validation=np.dot(valid_X ,theta) - valid_y
            loss_training=np.dot(train_X , theta) - train_y
            VLE = (0.5/(valid_X.shape[0])) * sum(np.dot(loss_validation.T, loss_validation))
            ERR = (0.5/(train_X.shape[0])) * sum(np.dot(loss_training.T,loss_training))
            validation_errors.append(abs(ERR-VLE))
            if abs(ERR-VLE) < dummy:
                final_theta = theta
                l2_final = l2
                dummy = abs(ERR-VLE)
        print("optimal value of lambda is " ,l2_final)
        print(final_theta)
        print(score(final_theta,test_X,test_y))
        plt.plot(L2_vals, validation_errors)
        plt.show()
        return final_theta,l2_final
main()
