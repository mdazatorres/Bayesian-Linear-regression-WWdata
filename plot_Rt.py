
import numpy as np
import pandas as pd
import matplotlib.dates as mdates
import matplotlib as mpl

import epyestim.covid19 as covid19
import matplotlib.pyplot as plt
from run_mcmc import mcmc_lr_ww, Params_ex


fontsize=20
plt.rcParams['font.size'] = fontsize

workdir = "./"
Pars = Params_ex()






def plot_RT(ax, label, color, colorh, ls, ht):
    init_training, end_training = Pars.training_date
    init, end = Pars.test_date

    mcmc = mcmc_lr_ww(init_training, end_training, init, end)
    city_data = mcmc.city_data[init:end]

    trace_Q500, trace_Q025, trace_Q975 = mcmc.linear_model_gibss(X=city_data['Ngene_norm_av7'])

    cases = np.exp(trace_Q500)
    date = city_data.index

    obs_cases = city_data.positives_raw.interpolate(method='linear', axis=0).ravel()

    davisdf_s = pd.Series(data=cases, index=date)
    davisdf_obs = pd.Series(data=obs_cases, index=date)

    ch_time_varying_r = covid19.r_covid(davisdf_s)
    ch_time_varying_r_obs = covid19.r_covid(davisdf_obs)


    dates_ = ch_time_varying_r.index.strftime("%b-%d-%y")
    dates_obs= ch_time_varying_r_obs.index.strftime("%b-%d-%y")
    #dates_=ch_time_varying_r.index

    mpl.rcParams['axes.spines.right'] = False
    lim=-15

    ax.plot(dates_,ch_time_varying_r['Q0.5'], color=color, ls=ls, lw=2)
    ax.fill_between(dates_, ch_time_varying_r['Q0.025'], ch_time_varying_r['Q0.975'], facecolor=color,
              alpha=0.3,hatch=ht,edgecolor=colorh,lw=1, label=label)

    ax.plot(dates_obs,ch_time_varying_r_obs['Q0.5'], color='k', ls=ls, lw=2)
    ax.fill_between(dates_obs, ch_time_varying_r_obs['Q0.025'], ch_time_varying_r_obs['Q0.975'], facecolor='k',
              alpha=0.3,hatch=ht,edgecolor='k',lw=1, label=label)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=1, interval=1))
    #ax.set_xlabel('2022')
    ax.tick_params(which='major', axis='x')
    ax.set_ylabel(r'$R_e(t)$ with 95% CI', fontsize=fontsize)

    ax.set_xlim([dates_[0],dates_[lim]])


    #ax.set_ylabel(r'$R_e(t)$ with 95% CI', fontsize=fontsize)

    ax.axhline(y=1, color="red")
    mpl.rcParams['hatch.linewidth'] = 1.8
    #ax.grid(linestyle='-')




fig, ax = plt.subplots(num=1, figsize=(12, 5))
plot_RT(ax, label='Cases', color='blue', colorh='blue', ls='-', ht=r'$\$')
#plot_comp_Rt(city, test_set=0, ax=ax)

#plot_comp_Rt_per(city=city, method=methods[1], ax=ax)

#davisdf_.index.name=None

