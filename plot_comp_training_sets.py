
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.pyplot import subplots
from scipy.stats import gamma
import matplotlib.dates as mdates
#from plot_Rt import plot_comp_Rt
import matplotlib as mpl
from datetime import date, timedelta
from run_mcmc import mcmc_lr_ww, Params_ex
from plot_training_set import fig_training_set
plt.rcParams['font.size'] = 17
font_xylabel = 17
font_leg =16

workdir = "./"
Pars = Params_ex()


def plot_linear_model(ax, cases=True, color='blue'):
    """ Plot estimated cases with the linear model
        :param city: (str)
        :param cases: (array)
        :params label:(str)
        :params test_set:(int)
        :params ax:(str)
        :return:fig: (fig):
    """
    init_training, end_training = Pars.training_date
    init,  end = Pars.test_date

    mcmc = mcmc_lr_ww(init_training, end_training, init, end)
    city_data = mcmc.city_data[init: end]
    #linear bayesian regresion ridge
    #X = np.log(city_data['Ngene_norm_av7'].values.reshape(-1, 1) + mcmc.eps)
    #ymean, ystd = mcmc.linear_model(X=X)
    trace_Q500, trace_Q025, trace_Q975 = mcmc.linear_model_gibss(X=city_data['Ngene_norm_av7'])

    ax.plot(city_data.index, np.exp(trace_Q500),  markersize=2, linewidth=2, color=color, label='Predicted')
    ax.fill_between(city_data.index, np.exp(trace_Q025), np.exp(trace_Q975), color=color, alpha=0.3)
    #ax.axvspan(init_training, end_training, alpha=0.2, color='k', label='Training period')

    if cases:
      ax.plot(city_data.index, city_data.positives_raw, 'o', markersize=4, linewidth=2, color='k', alpha=0.8,label="Observed")
      ax.plot(city_data.index, city_data.positives_average,  markersize=4, linewidth=2, color='r', alpha=0.8,
              label="Smoothed")

    #k=0.5
    #if ridge_reg == 0:
    #    ax.fill_between(city_data.index, np.exp(ymean - k * ystd), np.exp(ymean + k * ystd), facecolor=color, alpha=0.3, hatch= r'$\$', lw=1, edgecolor='purple',label='ridge bayesian')  # label="Predict std"

   # ax.plot(city_data.index, np.exp(ymean), color=color,lw=2)
    ax.set_xlim([city_data.index[0] - timedelta(days=1), city_data.index[-1] + timedelta(days=0)])
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=1, interval=1))
    ax.set_xlabel('2021                                 2022', loc='left',fontsize = font_xylabel)
    ax.set_ylabel('Cases', fontsize = font_xylabel)

    ax.tick_params(which='major', axis='x')
    ax.legend(fontsize=font_leg, loc='upper center')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    mpl.rcParams['hatch.linewidth'] = 1.8

    #if save:
    #    fig.savefig(workdir + 'figures/' +'linear_reg_pred' + '.jpg')




# def plot_rt(city, save):
#     fig, ax = subplots(num=1, figsize=(9, 5))
#     plot_comp_Rt(city,test_set=0, ax=ax)
#     fig.tight_layout()
#     if save:
#         fig.savefig(workdir + 'figures/'+ city + '_Rt' + '.jpg')

city='Davis'
#city = 'UCDavis'
#city = 'Woodland'
#plot_rt(city, save=True)
#comparison_conv_Tsets(city, output_name_1=None, output_name_2=None, workdir=workdir, save=True)
#comparison_linear_Tsets(city, save=True)
#plot_linear_vs_conv(city=city, test_set=0, output_name=None, workdir=workdir, save=True)
#plot_linear_model(city=city, cases=False, label='Linear', ax=ax)


def plot_lm_training_set(save):
    fig, _axs = plt.subplots(nrows=2, ncols=1, figsize=(10, 8), sharex=True)  # sharey=True
    # fig.subplots_adjust(right=0.5)
    fig.subplots_adjust(hspace=0.1)
    axs = _axs.flatten()

    init, end = Pars.test_date
    xy=[3,450]
    axs[0].text(pd.to_datetime(init)- timedelta(days=24), xy[0], 'A', ha='left', size='x-large', color='black',weight='bold')
    axs[1].text(pd.to_datetime(init)- timedelta(days=24), xy[1], 'B', ha='left', size='x-large', color='black',weight='bold')

    #fig, ax = subplots(num=1, figsize=(9, 5))
    fig_training_set(ax=axs[0])
    plot_linear_model(ax=axs[1], cases=True, color='blue')
    fig.tight_layout()

    if save:
        fig.savefig(workdir + 'figures/' +'linear_reg_pred' + '.jpg', dpi=300)

plot_lm_training_set(save=True)
# axs[0].text(mcmc.init0 - timedelta(days=42), xy[0], 'A', ha='left', size='xx-large', color='black',weight='bold')
# axs[1].text(mcmc.init0 - timedelta(days=42), xy[1], 'B', ha='left', size='xx-large', color='black',weight='bold')







