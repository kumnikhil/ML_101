import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse
from sklearn.cross_validation import train_test_split

def MoreData(N):
    f_x = lambda x:np.sin(1./x)
#    f_x = lambda x:np.sin(np.pi*x)*np.exp(-0.5*np.pi*x)
#    f_x = lambda x:np.sin(np.pi*x)*np.exp(x**2)
    # plotting the function 
    x_min,x_max = -0.1,0.1
#    x_min,x_max = -5.,5.
    np.random.seed(1234)
    X = np.random.uniform(low=x_min, high=x_max, size=(N,1))
    Y = f_x(X)
    return X,Y

def linear_performance(train_sizes,X_test,Y_test,model):
    mse_list = []
    for N_train in train_sizes:
        X_train,Y_train = MoreData(N_train)
        model.fit(X_train, Y_train)
        
        Y_pred_test = model.predict(X_test)
        mse_list.append(mse(Y_test,Y_pred_test))
    return mse_list


