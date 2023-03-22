from run_mcmc import mcmc_lr_ww, Params_ex
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.pyplot import subplots
import matplotlib.dates as mdates
from datetime import date, timedelta
import datetime


plt.rcParams['font.size'] = 18

workdir = "./"
Pars = Params_ex()

font_xylabel=18
font_leg=17

def fig_training_set(ax):
  """ Plot Testing vs  cases and the training-sets.
    :param city (str)
    :param save (bolean)
    :return: figure (.png)
  """
  init_training, end_training = Pars.training_date
  init_date, end_date = Pars.test_date
  mcmc = mcmc_lr_ww(init_training=init_training, end_training=end_training, init=init_date, end=end_date)

  city_data = mcmc.city_data
  city_data = city_data[init_date: end_date]
  city_data_w = city_data.groupby(pd.Grouper(freq="W")).sum()

  #fig, ax = subplots(num=1, figsize=(18, 8))

  #ax.semilogy(city_data_w.index, city_data_w['Testing'], '-o', markersize=3, linewidth=2, color='c', label='Tests')
  #ax.semilogy(city_data_w.index, city_data_w['positives'], '-o', markersize=3, linewidth=2,  color='k', label='Cases')


  city_data_w['m test'] = np.round(city_data_w['Testing'] / city_data_w['Testing'].shift(), 2)
  city_data_w['m pos'] = np.round(city_data_w['positives'] / city_data_w['positives'].shift(), 2)

  city_data_w['rate'] = city_data_w['m pos'] / city_data_w['m test']
  ax.axhline(y=1, xmin=0, xmax=1, ls='dashed', color='red')
  ax.plot(city_data_w.index, city_data_w['rate'], '-o', ms=2, lw=1.5, color='black', label=r'$r^p$')
  ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
  ax.set_xlim([city_data.index[0] - timedelta(days=1), city_data.index[-1] + timedelta(days=0)])
  ax.set_ylabel(r'$r^p$',fontsize=font_xylabel)

  ax.axvspan(init_training, end_training, alpha=0.2, color='gray', label='Training period')
  ax.xaxis.set_major_locator(mdates.DayLocator(interval=30))
  ax.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=1, interval=1))
  ax.legend(fontsize=font_leg)
  #ax.legend(frameon=False)
  ax.tick_params(which='major', axis='x')
  #ax.set_xlabel('2021                                 2022', loc='left',fontsize = font_xylabel)

  plt.setp(ax.get_xticklabels(), rotation=0, ha="center", rotation_mode="anchor")




#fig, ax = subplots(num=1, figsize=(9, 5))
#fig_training_set(ax)
#fig.tight_layout()
#fig.savefig(workdir + 'figures/' + 'training_periods' + '.jpg', dpi=600)
#city = 'Davis (sludge)'
#city = 'Davis'
#city = 'UCDavis'

#plot_selec_training_set(city='UCDavis', save=True)


