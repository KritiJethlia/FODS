from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
from get_data import get_data
from split_data import split_data
from standardize import *
from A3_final2 import main
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from math import sqrt
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
    return theta
def surface_plot():
    X,y=get_data('insurance.txt')
    x1=X[:,0]
    x2=X[:,1]
    x1, mean, std = standardize(x1)
    x2, mean, std = standardize(x2)
    a,b=np.meshgrid(x1,x2)
    #l1_vals,l2_vals=main()
    l1_vals=[0.5, 0.30000000000000004, 0.30000000000000004, 0.1, 0.9, 0.1, 0.5, 0.30000000000000004, 0.9, 0.30000000000000004]
    l2_vals=[0.1, 0.1, 0.30000000000000004, 0.9, 0.7000000000000001, 0.30000000000000004, 0.30000000000000004, 0.5, 0.5, 0.1]
    lr=0.000001
    for deg in range(10,11):
        poly_features=PolynomialFeatures(degree=deg)
        X_poly=poly_features.fit_transform(X)
        y=y*(1/10000)
        X, mean, std = standardize(X_poly)
        X[:,0]=1
        m = X.shape[0]
        n = X.shape[1]
        print("-----------Degree is   ",deg)
        '''Plot for gradient descent
        theta = np.random.rand(n, 1)
        theta= train_model(X, y, theta, lr)
        y_predict=np.dot(X,theta)
        Z = y_predict.reshape(1,-1)
        fig = plt.figure()
        ax = fig.add_subplot(111,projection = '3d')
        cset=ax.plot_surface(a, b, Z,alpha=0.5, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        ax.clabel(cset, fontsize=9, inline=1)
        plt.show()
        #----Plot for L1------
        print("Plot for L1 regularization")
        theta_L1=L1_reg_weights(lr,X,y,l1_vals[deg-1])
        y_predict=np.dot(X,theta_L1)
        Z = y_predict.reshape(1,-1)
        fig = plt.figure()
        ax = fig.add_subplot(111,projection = '3d')
        cset=ax.plot_surface(a, b, Z,alpha=0.5, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        ax.clabel(cset, fontsize=9, inline=1)
        plt.show()
        '''
        ''' Plot for L2'''
        print("Plot for L2 regularization")
        theta_L2=L2_reg_weights(lr,X,y,l2_vals[deg-1])
        y_predict=np.dot(X,theta_L2)
        Z = y_predict.reshape(1,-1)
        fig = plt.figure()
        ax = fig.add_subplot(111,projection = '3d')
        cset=ax.plot_surface(a, b, Z,alpha=0.5, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        ax.clabel(cset, fontsize=9, inline=1)
        plt.show()
        
def L1_reg_weights(lr, X,y,l1):
    dummy=1e10
    m = X.shape[0]
    n = X.shape[1]
    fun = lambda x: (x / abs(x))
    theta = np.random.rand(n,1)
    previous_error = 1e10
    for i in range(50000):
        loss=np.dot(X,theta)-y
        err = 0.5 * (np.dot(loss.T, loss) + l1*sum([w*w for w in theta]))
        abs_w = np.array([fun(w) for w in theta])
        theta -= lr * ((np.dot((X.T),loss))) + lr*(0.5*l1*abs_w)
        if abs(previous_error-err) < 5e-3:
                break
        previous_error = err
    return theta
def L2_reg_weights(lr, X,y,l2):
        m = X.shape[0]
        n = X.shape[1]
        previous_error = 1e10
        theta = np.random.rand(n,1)
        for i in range(100000):
            loss=np.dot(X,theta)-y
            err = 0.5 * (np.dot(loss.T, loss) + l2*sum([w*w for w in theta]))
            theta -= lr * ((np.dot((X.T),loss))) + lr*(l2*theta)
            if abs(previous_error-err) < 5e-3:
                break
            previous_error = err
        return theta