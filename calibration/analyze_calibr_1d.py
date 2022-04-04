# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 13:03:42 2020

@author: 03125327
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import emcee

#%%
def sample_from_posterior(mcmc_flat_chain, n_samples, mcmc_logprob):
    """
    Takes a number of parameter samples from the posterior distribution, and computes
    the hydrology with them.
    Useful for checking model performance on the testing set (i.e., on runs over
    data not used for parameter calibration). For instance, for approximating
    statistics of confidence intervals and the like.
        - 
    Parameters
    ----------
    mcmc_flat_chain : numpy array
        MCMC chain, flattened. What you get when running emcee.get_chain(flat=True)
    n_samples : int
        Number of samples to take from the posterior and, thus, number of times
        to run the hydro
    mcmc_logprob : numpy array
        MCMC log probabilities. What you get when running emcee.get_log_prob(flat=True)

    Raises
    ------
    ValueError
        When more samples want to be drawn than there exist in the posterior

    Returns
    -------
    list
        Samples from the posterior without infinities

    """

    if n_samples > len(mcmc_flat_chain):
        raise ValueError("There are not enough samples in the MCMC to draw from.")
    
    # choose random samples from posterior that dont have posterior=-inf
    randints = []
    while len(randints)<100:    
        rint = np.random.randint(low=0, high=len(mcmc_flat_chain))
        if np.isfinite(mcmc_logprob[rint]):
            randints.append(rint)
        else:
            continue
        
    post_params = [mcmc_flat_chain[ind] for ind in randints]
    
    return post_params

def confidence_intervals(ci_percents, n_sensors, ndays, n_posterior_samples):
    """
    Compute CI from posterior-sampled-hydro.
    TODO: Not a definitive setup with the dimensions of the arrays, couldn't think
    of  one that was generalizable, has to be tailored for each case.

    Parameters
    ----------
    ci_percents : list
        List of confidence intervals to compute
    n_sensors : int
        Number of sensors
    ndays : int
        ndays the hydro model has run

    Returns
    -------
    conf_interv_results : numpy array
        Call as follows: conf_interv_results[which sensor?][which day?][which CI?]
        From there: -> [0] is maximum of the CI
                        [1] is minimum of the CI

    """

    # translate CI percentages to how many items to pick up from array
    n_elements_to_pick = np.array(ci_percents)/100.*n_posterior_samples 
    n_elements_to_pick = n_elements_to_pick.astype(int)
    
    conf_interv_results = np.zeros(shape=(n_sensors, ndays, len(ci_percents), 2))
    for i in range(n_sensors):
        modelled = results[:,i]
        diff = np.abs(modelled - measurements.T[i])
        for day in range(NDAYS):
            model_day = modelled[:, day]
            diff_day = diff[:, day]
            model_diffsorted = model_day[diff_day.argsort()] # modelled values sorted by absolute difference
            for n, nele in enumerate(n_elements_to_pick):
                conf_interv_results[i,day,n][0] = np.max(model_diffsorted[:nele])
                conf_interv_results[i,day,n][1] = np.min(model_diffsorted[:nele])
    
    return conf_interv_results
    


def read_from_backend(filename):
    
    reader = emcee.backends.HDFBackend(filename, read_only=True)
    samples = reader.get_chain(discard=0, thin=1, flat=True)
    print(f"Read from backend. \n   Samples shape = {samples.shape}")
    logprobs = reader.get_log_prob(flat=True)
    frac_infinities = 1 - np.count_nonzero(np.isfinite(logprobs))/samples.shape[0]
    print(f"   Fraction of infinities = {frac_infinities * 100:.2f}%")
    
    return reader

def corner_plot(samples, labels, savefig=True):
    import corner
    fig = corner.corner(samples, labels=labels)
    if savefig:
        fig.savefig("1d_cal_output/MCMC_corner_result.png")
    
    return 0

#%%
    
#fname = r"C:\Users\03125327\github\blopti_dev\calibration\mcmc_result_chain.h5"
fname = r"1d_cal_output\mcmc_result_chain_FC_NEW.h5"

reader = read_from_backend(fname)
flat_samples = reader.get_chain(flat=True)
fat_samples = reader.get_chain(flat=False)

#%% 
# Take a look at walkers and autocorrelation time

nparams = fat_samples.shape[2]
fig, axes = plt.subplots(nparams, sharex=True)

axes[0].set_title('Before discarding and thinning')

labels = ["s1", "s2", 't1', 't2']
for i in range(nparams):
    ax = axes[i]
    ax.plot(fat_samples[:, :, i], "k", alpha=0.3)
    ax.set_xlim(0, len(fat_samples))
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("step number");

tau = reader.get_autocorr_time()
print(tau)

# Even if logprobs give -inf, show values of parameters explored
# s1s = flat_samples[:,0]
# s2s = flat_samples[:,1]
# t1s = flat_samples[:,2]
# t2s = flat_samples[:,3]

# plt.figure()
# plt.plot(s2s, 'o', ms=1 ,label='s1')
# plt.plot(s2s, 'o', ms=1 ,label='s2')
# plt.plot(t1s, 'o', ms=1 ,label='t1')
# plt.plot(t2s, 'o', ms=1 ,label='t2')

# plt.legend()

#%%
# Thin and discard samples
DISCARD = 10000
THIN = 1000# 1=no thinning
flat_samples = reader.get_chain(discard=DISCARD, thin=THIN, flat=True)
fat_samples = reader.get_chain(discard=DISCARD, thin=THIN, flat=False)
print(f"flat samples shape: {flat_samples.shape}")

#%%
# Corner plot
corner_plot(flat_samples, labels, savefig=True)

#%%
# ML and ranges for values
flat_samples_all = reader.get_chain(flat=True)
log_probs_all = reader.get_log_prob(flat=True)
log_probs = reader.get_log_prob(flat=True, discard=DISCARD, thin=THIN)
max_lp = np.max(log_probs_all)
max_likelihood_params = flat_samples_all[np.where(log_probs_all==max_lp)[0]][0]

print(f'maximum likelihood params: {max_likelihood_params}')
   

#%%
"""
 Plot Sy and T. Compute and plot confidence intervals.
"""
def S(zeta, s1, s2):
    # in terms of zeta = h - dem
    return s1*np.exp(s2*zeta)

def T(zeta, b, t1, t2):
    # in terms of zeta = h - dem
    return t1 * (np.exp(t2*zeta) - np.exp(-t2*b))

def turn_conf_intervals_into_array_indices(conf_intervals_list, flat_samples):
    ci_array = np.array(conf_intervals_list)
    ci_indices = ci_array * len(flat_samples) / 100
    return ci_indices.astype(dtype=int)

def sort_samples_by_logprobs(flat_samples, log_probs):
    return flat_samples[log_probs.argsort()[::-1]]

def max_and_min_sto_and_trans_curves(sto_array, tra_array):
    sto_ci = [0] * len(CONF_INTERVALS_LIST)
    tra_ci = [0] * len(CONF_INTERVALS_LIST)
    for i, ci in enumerate(conf_int_indices_to_pick):
        # storage
        sto_arr_ci = sto_array[:ci]
        sto_max_ci = sto_arr_ci.max(axis=0)
        sto_min_ci = sto_arr_ci.min(axis=0)
        sto_ci[i] = [sto_min_ci, sto_max_ci]
        # transmissivity
        tra_arr_ci = tra_array[:ci]
        tra_max_ci = tra_arr_ci.max(axis=0)
        tra_min_ci = tra_arr_ci.min(axis=0)
        tra_ci[i] = [tra_min_ci, tra_max_ci]
        
    return sto_ci, tra_ci

def plot_all_sto_and_tra(sto_array, tra_array):
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(15,10))
    axS = ax[0]; axT = ax[1]
    axS.set_title('Specific yield'); axT.set_title('Transmissivity')
    axS.set_xlim(left=0, right=10); axT.set_xlim(left=0, right=100)
    axS.set_xlabel('$S_y(\zeta)$'); axS.set_ylabel('$\zeta (m)$')
    axT.set_xlabel('$T(\zeta) m^2d^{-1}$'); axT.set_ylabel('$\zeta (m)$')
    
    for s, t in zip(sto_array, tra_array):
        # axS.plot(s, h_grid, alpha=1/(30*np.log(flat_samples.shape[0])), color='black')
        # axT.plot(t, h_grid, alpha=1/(30*np.log(flat_samples.shape[0])), color='black')
        axS.plot(s, ZETA_GRID, alpha=1/256, color='black')
        axT.plot(t, ZETA_GRID, alpha=1/256, color='black')
    
    axS.hlines(y=0, xmin=0, xmax=10, colors='brown', linestyles='dashed', label='peat surface')
    axT.hlines(y=0, xmin=0, xmax=100, colors='brown', linestyles='dashed', label='peat surface')
    
    plt.show()
    
    return None

def plot_conf_intervals_and_ML(max_and_min_sto_curves, max_and_min_tra_curves):
    
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(15,10))
    # CI
    axS = ax[0]; axT = ax[1]
    axS.set_xlim(left=0, right=10); axT.set_xlim(left=0, right=100)
    axS.set_xlabel('$S_y(\zeta)$'); axS.set_ylabel('$\zeta (m)$')
    axT.set_xlabel('$T(\zeta) m^2d^{-1}$'); axT.set_ylabel('$\zeta (m)$')
    
    for i, ci in enumerate(CONF_INTERVALS_LIST):
        axS.fill_betweenx(ZETA_GRID, max_and_min_sto_curves[i][1], max_and_min_sto_curves[i][0], label= f'{ci}% conf. int.', alpha=0.1)
        axT.fill_betweenx(ZETA_GRID, max_and_min_tra_curves[i][1], max_and_min_tra_curves[i][0], label= f'{ci}% conf. int.', alpha=0.1)
    
    
    axS.plot(S(ZETA_GRID, max_likelihood_params[0], max_likelihood_params[1]), ZETA_GRID, label='ML')
    axT.plot(T(ZETA_GRID, B,  max_likelihood_params[2], max_likelihood_params[3]), ZETA_GRID, label='ML')
    
    fig.legend()
    
    axS.hlines(y=0, xmin=0, xmax=10, colors='brown', linestyles='dashed', label='peat surface')
    axT.hlines(y=0, xmin=0, xmax=100, colors='brown', linestyles='dashed', label='peat surface')
    
    plt.show()
    
    raise Warning('Confidence intervals are not right. Review this.')
    
    return None
    
    
# Parameters   
B = 8.0 # impermeable bottom elevation in m. Parameter to be chosen.
CONF_INTERVALS_LIST = [90, 68, 10] # confidence interval %
ZETA_GRID = np.arange(-8, 1, 0.01)

CONF_INTERVALS_LIST = sorted(CONF_INTERVALS_LIST) #For the plot

conf_int_indices_to_pick = turn_conf_intervals_into_array_indices(CONF_INTERVALS_LIST, flat_samples)
sorted_flatsamples =  sort_samples_by_logprobs(flat_samples, log_probs)

# Arrays containing storage and transm for all parameters
sto_array = np.array([S(ZETA_GRID, params[0], params[1]) for params in sorted_flatsamples])
tra_array = np.array([T(ZETA_GRID, B, params[2], params[3]) for params  in sorted_flatsamples])

# Plot all curves with some alpha    
plot_all_sto_and_tra(sto_array, tra_array)

max_and_min_sto_curves, max_and_min_tra_curves = max_and_min_sto_and_trans_curves(sto_array, tra_array) 
# Structure of arrays: [[Smin_ci1, Smax_ci1], [Smin_ci2, Smax_ci2], ...], and each one ranges for all ZETA_GRID values

# Plot confidence intervals and ML curves for S and T
plot_conf_intervals_and_ML(max_and_min_sto_curves, max_and_min_tra_curves)


#%%     
"""
 Sample from resulting posterior.
 TODO: CHANGE THESE FOR NEW HYDRO ETC.
"""
def xxxxxxxxxxxx(samples, logprobs):
    N_POSTERIOR_SAMPLES = 10
    post_params = sample_from_posterior(samples, N_POSTERIOR_SAMPLES, logprobs)
                          
    #%%
    """
    Run hydro with set of params from the posterior
    """
    import calibration_1d
    
    bcleft, bcright, measurements, days, precip, evapotra = calibration_1d.read_sensors('fabricated_data.txt')
    boundary = np.array(list(zip(bcleft, bcright)))
    HINI = 8.; nx = 100; dx = 1.; dt = 1.; NDAYS = len(measurements);
    SENSOR_LOCATIONS = [0, 12, 67, 99];
    
    results = []
    for params in post_params:
        s0 = params[0]; s1 = params[1] 
        t0 = params[2]; t1 = params[3]; t2 = params[4];
        
        theta_ini = np.exp(s0 + s1*HINI) 
        try:
            simulated_wtd = calibration_1d.hydro_1d(nx, dx, dt, params, theta_ini,
                                                    NDAYS, SENSOR_LOCATIONS, boundary, precip, evapotra)
        except:
            print("PROBLEM IN HYDRO WITH PARAMS")
            continue
        
        results.append(simulated_wtd.T) # NOTE, appending the transpose!!
      
    results = np.array(results)
    #%%
    """
     Compute confidence intervals
    """
    ci_percents = [20, 50, 90]
    n_sensors = len(SENSOR_LOCATIONS)
    conf_interv_results = confidence_intervals(ci_percents, n_sensors, NDAYS, N_POSTERIOR_SAMPLES)
    
    
    #%%
    """
     Plot over data, sensor by sensor:
       - samples from posterior WTD over data
       - Confidence intervals
    """
                
    fig,axes = plt.subplots(nrows=2, ncols=2)
    axes = axes.ravel()
    
    fig_ci, axes_ci = plt.subplots(nrows=2, ncols=2)
    axes_ci = axes_ci.ravel()
    
    for sensor_n, sensor_loc in enumerate(SENSOR_LOCATIONS):
        sensor_simulated_wtd = results[:,sensor_n] # order results by sensors 
        measured_value_in_sensor = measurements.T[sensor_n]
        days_x_axis = list(range(0,len(measured_value_in_sensor)))
        for simu_sensor in sensor_simulated_wtd:
            axes[sensor_n].plot(days_x_axis, simu_sensor, 'orange', alpha=0.1)
        
        # confidence intervals
        colors = ['blue', 'red', 'yellow']
        if len(colors) < len(ci_percents):
            raise ValueError("Too few colors")
        for nci, ci in enumerate(ci_percents):
            top_ci = conf_interv_results[sensor_n][:,nci][:,0]
            bottom_ci = conf_interv_results[sensor_n][:,nci][:,1]
            axes_ci[sensor_n].fill_between(days_x_axis, top_ci, bottom_ci,
                                           label=f"{ci}% CI", alpha=0.2, color=colors[nci])
            axes_ci[sensor_n].plot(days_x_axis, top_ci, color=colors[nci], linewidth=0.2)
            axes_ci[sensor_n].plot(days_x_axis, bottom_ci, color=colors[nci], linewidth=0.2)
        
            
        
        # true, measured values    
        axes[sensor_n].plot(days_x_axis, measured_value_in_sensor, 'black')
        axes_ci[sensor_n].plot(days_x_axis, measured_value_in_sensor, 'black')
        
        axes[sensor_n].set_title(f"sensor_n {sensor_n} at location {sensor_loc}")
    
    fig_ci.legend()
    plt.show()
            