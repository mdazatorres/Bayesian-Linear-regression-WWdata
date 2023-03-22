import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import BayesianRidge
from scipy.stats import multivariate_normal, invgamma
from gibbs_sampler_linear_reg import Gibbs_sampler_lr
import matplotlib.pyplot as plt


class Params_ex:
    """
    Set training sets, the study period, shedding length(m) and additional params to plot
    """
    def __init__(self):
        self.training_date = ['2022-01-16', '2022-02-14'] #[['2022-01-09', '2022-02-05'], ['2021-12-12', '2022-01-08']]
        #self.training_date = ['2022-01-09', '2022-02-05']
        #self.test_date =  ['2021-12-01', '2022-03-31']
        self.test_date = ['2021-12-01', '2022-07-01']

        #self.ylim ={'Davis':[0.0035, 400], 'UCDavis':[0.0022, 400]}
        #self.colors = ['green', 'red','yellow']
        self.colors = [[0.19215686, 0.50980392, 0.74117647], [0.87058824, 0.17647059, 0.14901961],[0.41568627, 0.31764706, 0.63921569] ]

class mcmc_lr_ww:
    def __init__(self, init_training, end_training, init, end):
        self.data_ww = pd.read_csv('data/data_ww_cases.csv', index_col=0)
        self.city = 'Davis'
        self.init_training = init_training
        self.end_training = end_training
        self.init=init
        self.end= end
        self.size_window = 7  # for smothing ww data
        self.read_data()
        self.eps = 1e-10

        # Initial conditions for Beta, and s2
        self.Beta_0 = np.array([0, 1])
        self.s2_0 = 1.5
        # Hyper parameters
        # Normal distribution parameters
        self.Mu = np.array([5, 1])
        self.Sig = np.diag([3 ** 2, 1 ** 2])  # [sx2, sy2]
        # Inverse Gamma parameters
        self.A = 2
        self.B = 1  # Var(Y) , 1.5


    def read_data(self):
        self.city_data = self.data_ww[self.data_ww['City'] == self.city]
        self.city_data = self.city_data.reset_index()
        self.city_data.loc[self.city_data.Testing.isna(), 'positives_raw'] = np.nan
        self.city_data['positives_average'] = np.copy(self.city_data['positives_raw'].rolling(window=7, center=True, min_periods=3).mean())
        self.city_data['Ngene_norm_av7'] = self.city_data['Ngene_norm'].rolling(window=self.size_window, center=True, min_periods=3).mean()

        self.Data_train = self.city_data[(self.city_data['Date'] >= self.init_training) & (self.city_data['Date'] <= self.end_training)]
        self.Data_test= self.city_data[(self.city_data['Date'] >= self.init) & (self.city_data['Date'] <= self.end)]

        self.Xtest = self.Data_test['Ngene_norm_av7']
        self.Xtrain = self.Data_train['Ngene_norm_av7']

        #X= city_data['Ngene_norm_av7']

        self.Ytrain = self.Data_train.positives_average
        self.city_data.index = pd.DatetimeIndex(self.city_data['Date'])



    def linear_model(self,X):
        """
        Compute regression model using Bayesian Ridge
        """
        eps = 1e-10

        x_train = np.log(self.Xtrain.values.reshape(-1, 1)+ eps)
        y_train = np.log(self.Ytrain + eps)

        reg = BayesianRidge(tol=1e-10, fit_intercept=True, compute_score=True)
        reg.fit(x_train, y_train)
        ymean, ystd = reg.predict(X, return_std=True)

        return ymean, ystd

    def linear_model_gibss(self,X):
        X = np.log(X)
        X = X.values.reshape(-1, 1)
        X = np.hstack((np.ones(X.shape), X))

        x_train = np.log(self.Xtrain.values.reshape(-1, 1) + self.eps)
        y_train = np.log(self.Ytrain + self.eps)

        X_train = np.hstack((np.ones(x_train.shape), x_train))
        #X = np.hstack((np.ones(X.shape), X))

        self.gs = Gibbs_sampler_lr(Beta_0=self.Beta_0, s2_0=self.s2_0, Mu=self.Mu, Sig=self.Sig, A=self.A, B=self.B, Xtrain=X_train, Ytrain=y_train)
        self.gs.Gibbs_Run()
        trace_Q500, trace_Q025, trace_Q975 = self.gs.y_pred(X)
        return trace_Q500, trace_Q025, trace_Q975




# Pars = Params_ex()
# city = 'Davis'
#
# init_training, end_training = Pars.training_date
# init, end = Pars.test_date
#
# mcmc = mcmc_lr_ww(city, init_training, end_training, init, end)
#
# trace_Q500, trace_Q025, trace_Q975 = mcmc.linear_model_gibss(mcmc.Xtest)
#
# city_data_= mcmc.Data_test
#
#
# from matplotlib.pyplot import subplots
# fig, ax = subplots(num=1, figsize=(9, 5))
#
# ax.plot(city_data_.index, city_data_.positives_average, 'o', color='k' )
#
# ax.plot(city_data_.index, np.exp(trace_Q500), color='gray')
# ax.plot(city_data_.index, np.exp(trace_Q025), color='blue')
# ax.plot(city_data_.index, np.exp(trace_Q975), color='blue')

