 
#### Simulation codes for the examples presented in:
### Mathematical Modelling and Wastewater-Based Epidemiology: Public Health Applications

This module consists in the following files:

#### Data
data_ww_cases.csv
Testing_case_HYT.xls

#### Codes

#### gibbs_sampler_linear_reg.py
This class contain a Bayesian linear regression using a Gibbs sampler.

#### run_mcmc.py
Main code:
- Set all the parameters for the models
- Load and processed data
- Likelihood, priors for mcmc are defined
- Linear model 

#### plot_training_set.py
We plotted number of tests administered and cases by week for Davis. Also the training periods  used for the analysis.


#### plot_data.py
To plot ww concentration and covid-19 cases (trimmed and moving average). 

#### plot_comp_training_sets.py

- plot_linear_model(): Plot estimated cases with the linear model







