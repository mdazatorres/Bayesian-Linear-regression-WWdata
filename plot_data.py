from matplotlib.pyplot import subplots
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
#from run_mcmc import mcmc_conv, Params
import pandas as pd
from datetime import date, timedelta
from run_mcmc import mcmc_lr_ww, Params_ex
import numpy as np

plt.rcParams['font.size'] = 16
fontsize = 16
font_leg=16
Pars = Params_ex()
workdir = "./"

def plot_data_test(save):

    init_training, end_training = Pars.training_date
    init, end = Pars.test_date
    mcmc = mcmc_lr_ww(init_training, end_training, init, end)
    city_data = mcmc.city_data


    city_data = city_data[(city_data.index >= init) & (city_data.index <= end)]
    Date = pd.DatetimeIndex(city_data['Date'])

    fig, ax = plt.subplots(figsize=(13, 5))
    fig.subplots_adjust(right=0.75)
    twin1 = ax.twinx()
    twin2 = ax.twinx()
    twin2.spines.right.set_position(("axes", 1.1))
    Y = city_data['Ngene_norm']; lab1= 'N/PMMoV'
    Y1 = city_data['positives']; lab2='Cases'
    Y2 = city_data['Testing']; lab3='Tests'
    p2 = twin1.bar(Date, Y1, color='green', alpha=0.4, label=lab2)
    p3 = twin2.bar(Date, Y2, color='b',alpha=0.2, label=lab3)
    p1, = ax.plot(Date, Y, ".", color='k', label=lab1)

    ax.set_xlim(Date[0]-timedelta(days=2), Date[-1])

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=1, interval=1))
    # ax.tick_params(which='major', axis='x')
    yl = 1.02
    ax.set_ylim(0, np.nanmax(Y) * yl)
    twin1.set_ylim(0, np.nanmax(Y1) * yl)
    twin2.set_ylim(0, np.nanmax(Y2) * yl)

    ax.set_ylabel(lab1, fontsize=fontsize)
    twin1.set_ylabel(lab2, fontsize=fontsize)
    twin2.set_ylabel(lab3, fontsize=fontsize)

    tkw = dict(size=4, width=1.5)
    ax.tick_params(axis='x', **tkw)
    ax.legend(handles=[p1, p2, p3], loc='upper center', fontsize=font_leg)

    ax.set_xlabel('2021                                             2022', loc='left', fontsize=fontsize)
    ax.grid(color='gray', linestyle='--', alpha=0.2)
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center", rotation_mode="anchor")
    fig.tight_layout()

    if save:
        fig.savefig(workdir + 'figures/' +'cases_vs_tests.jpg',dpi=300)


def plot_data_ave(save):

    init_training, end_training = Pars.training_date
    init, end = Pars.test_date
    mcmc = mcmc_lr_ww(init_training, end_training, init, end)
    city_data = mcmc.city_data
    city_data = city_data[(city_data.index >= init) & (city_data.index <= end)]
    Date = pd.DatetimeIndex(city_data['Date'])

    fig, ax = plt.subplots(figsize=(13, 5))
    fig.subplots_adjust(right=0.75)
    twin1 = ax.twinx()
    twin2 = ax.twinx()
    twin2.spines.right.set_position(("axes", 1.09))
    Y = city_data['Ngene_norm_av7']; lab1= 'N/PMMoV'
    Y1 = city_data['positives_average']; lab2='Cases'
    Y2 = (city_data['positives_raw']/city_data['Testing']).rolling(window=7, center=True, min_periods=3).mean()
    lab3='Test positivity rate'
    p2, = twin1.plot(Date, Y1, color='green',  label=lab2)
    p3, = twin2.plot(Date, Y2, color='b',label=lab3)
    p1, = ax.plot(Date, Y,  color='k', label=lab1)

    ax.set_xlim(Date[0]-timedelta(days=2), Date[-1])

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=1, interval=1))
    # ax.tick_params(which='major', axis='x')
    yl = 1.02
    ax.set_ylim(0, np.nanmax(Y) * yl)
    twin1.set_ylim(0, np.nanmax(Y1) * yl)
    twin2.set_ylim(0, np.nanmax(Y2) * yl)

    ax.set_ylabel(lab1, fontsize=fontsize)
    twin1.set_ylabel(lab2, fontsize=fontsize)
    twin2.set_ylabel(lab3, fontsize=fontsize)

    tkw = dict(size=4, width=1.5)
    ax.tick_params(axis='x', **tkw)
    ax.legend(handles=[p1, p2, p3], loc='upper center', fontsize=font_leg)
    ax.set_xlabel('2021                                             2022', loc='left', fontsize=fontsize)
    ax.grid(color='gray', linestyle='--', alpha=0.2)
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center", rotation_mode="anchor")
    fig.tight_layout()

    if save:
        fig.savefig(workdir + 'figures/' +'cases_pr_conc.jpg',dpi=300)





city= 'Davis'
#plot_data_ave(save=True)
plot_data_test(save=True)