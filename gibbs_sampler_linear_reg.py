import numpy as np
from scipy.stats import multivariate_normal, invgamma,norm
import matplotlib.pyplot as plt


class Gibbs_sampler_lr:
    def __init__(self, Beta_0, s2_0, Mu, Sig, A, B, Xtrain, Ytrain):
        """"
        X=[1..,X,..]
        Y = X Beta + N(0, s2), Beta = (beta0, beta1)
        Beta ~ N(Mu, Sig), Sig =diag([S_0, S_1]), Mu=(mu_0, mu_1)
        s2 ~ InvGamma(A, B)
        Prior parameters
        Mu, Sig
        A, B
        Beta0: initial condition for Beta =[beta0_0, beta1_0]
        s2_0: initial condition for Beta =[beta0_0, beta1_0]

        """
        self.Beta_0 = Beta_0
        self.s2_0 = s2_0
        self.Mu = Mu
        self.Sig = Sig
        self.A = A
        self.B = B
        self.Xtrain = Xtrain
        self.Ytrain = Ytrain
        self.n_sample = 20000
        self.T=5000


    def Gibbs_Run(self):
        self.Beta_sim = self.Beta_0.reshape(1, -1)
        self.sigma_sim = np.array([self.s2_0])

        Sig_inv = np.linalg.inv(self.Sig)
        prod = Sig_inv @ self.Mu
        XTX = self.Xtrain.T @ self.Xtrain
        XTY = self.Xtrain.T @ self.Ytrain
        n = len(self.Ytrain)
        A_ = n / 2 + self.A
        for i in range(self.n_sample):
            s2_0 = self.sigma_sim[-1]
            Sig_ = np.linalg.inv(Sig_inv + s2_0 ** (-1) * XTX)
            Mu_ = Sig_ @ (prod + s2_0 ** (-1) * XTY)

            Beta_new = multivariate_normal.rvs(Mu_, Sig_)

            self.Beta_sim = np.append(self.Beta_sim, Beta_new.reshape(1, -1), axis=0)

            mu = self.Xtrain @ Beta_new #Beta_sim[-1]
            B_ = 0.5*(self.Ytrain - mu).T @ (self.Ytrain - mu) + self.B

            s2_new = invgamma.rvs(a=A_, scale= B_) # = InvChi2(vb_N,Sb_N)
            self.sigma_sim = np.append(self.sigma_sim, s2_new)

        self.Theta = np.concatenate([self.Beta_sim, self.sigma_sim.reshape(-1, 1)], axis=1)
        return self.Theta


    def post_pred(self, X):
        n_pred = X.shape[0]
        i=0
        Output_trace = np.zeros((self.T, n_pred))
        while i < self.T:
            theta = self.Theta[-(i + 1)]
            beta = theta[:2]; s2 = theta[-1]
            #beta = self.Beta_sim[-(i + 1)]
            #s2 = self.sigma_sim[-(i + 1)]
            Output_trace[i] = X @ beta + norm.rvs(0, np.sqrt(s2),size=n_pred)
            i=i+1
        return  Output_trace

    def y_pred(self, X):
        Output_trace = self.post_pred(X)
        trace_Q500 = np.quantile(Output_trace, 0.5, axis=0)
        trace_Q025 = np.quantile(Output_trace, 0.025, axis=0)
        trace_Q975 = np.quantile(Output_trace, 0.975, axis=0)
        return trace_Q500, trace_Q025, trace_Q975


