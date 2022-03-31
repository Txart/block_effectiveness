#%%
import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from pathlib import Path
import pandas as pd
from multiprocessing import Pool
import emcee
import argparse
import platform
from scipy.interpolate import interp1d
from scipy.stats import zscore
import sys


# Own functions from same folder
from hydro_python_subroutine import solve_transient_both_diri
import calibration_1d_utilities

#%% Folders and additional imports

# Folders
data_parent_folder = Path('../data')
output_parent_folder = Path('1d_cal_output')

sys.path.insert(1, '..') # add parent folder to path
import estimate_daily_et
import evapotranspiration_fao
import get_data


#%% Parse arguments
if platform.system() == 'Linux': # Working on CSC machines
    parser = argparse.ArgumentParser(description='Run MCMC parameter estimation')
    
    parser.add_argument('--ncpu', default=1, help='(int) Number of processors', type=int)
    parser.add_argument('-cl','--chainlength', default=10, help='(int) Length of MCMC chain', type=int)
    parser.add_argument('-w','--nwalkers', default=8, help='(int) Number of walkers in parameter space', type=int)
    args = parser.parse_args()
    
    N_CPU = args.ncpu
    MCMC_STEPS = args.chainlength
    N_WALKERS = args.nwalkers

elif platform.system() == 'Windows':
    N_CPU = 1
    MCMC_STEPS = 10
    N_WALKERS = 8

#%% Functions
def h_from_zeta(zeta, dem):
    return zeta + dem

def zeta_from_h(h, dem):
    return h - dem

def h_from_theta(theta, dem, s1, s2):
    return np.log(s2/s1 * np.exp(s2*dem) * theta + 1)/s2

def theta_from_h(h, dem, s1, s2):
    return s1/s2 * (np.exp(s2*(h - dem)) - np.exp(-s2*dem))

#%% General stuff
patrol_post_names = ['Buring', 'Lestari', 'MainCamp', 'Ramin', 'Serindit', 'Sialang']
abbreviation_to_name = {'LE':'Lestari', 'MC': 'MainCamp', 'RA': 'Ramin','SE': 'Serindit', 'SI': 'Sialang'}

# %% Params
NX = 50 # 1d mesh number of cells
transect_length = 150 # metres
dx = transect_length / NX
dt = 1 # day
N_PARAMS = 4
b = 8 # peat depth in the transect in metres

MAX_INTERNAL_NITER = 10000
WEIGHT = 0.1
REL_TOL = 1e-7
ABS_TOL = 1e-7
SENSOR_MEASUREMENT_ERR = 0.05 # metres. Theoretically, 1mm

#%% Read data
# Read historic WTD sensor data from file
df_rain = pd.read_csv(Path.joinpath(data_parent_folder, 'Raw csv/Rain2021.csv'), sep=';')
df_relhum = pd.read_csv(Path.joinpath(data_parent_folder, 'Raw csv/RelHum2021.csv'), sep=';')
df_maxtemp = pd.read_csv(Path.joinpath(data_parent_folder, 'Raw csv/Temp_Max2021.csv'), sep=';')
df_mintemp = pd.read_csv(Path.joinpath(data_parent_folder, 'Raw csv/Temp_Min2021.csv'), sep=';')
df_avgtemp = pd.read_csv(Path.joinpath(data_parent_folder, 'Raw csv/Temp_Average2021.csv'), sep=';')
df_windspeed = pd.read_csv(Path.joinpath(data_parent_folder, 'Raw csv/WindSpeed2021.csv'), sep=';')

# rename "Main Camp" to "MainCamp"
for df in [df_rain, df_relhum, df_maxtemp, df_mintemp, df_avgtemp, df_windspeed]:
    df.rename(columns={'Main Camp': 'MainCamp'}, inplace=True)

# Change units in rain from mm/day to m/day
df_rain[patrol_post_names] = df_rain[patrol_post_names] * 0.001

# %% Fill in missing values
def fill_weather_dataframe_missing_values_with_reference_patrol_post(dataframe, reference_patrol_post_name):
    # Fill  missing values with reference station's recordings
    for name in set(patrol_post_names) - set([reference_patrol_post_name]):
        for day in range(365):
            if np.isnan(dataframe.loc[day, name]):
                dataframe.at[day, name] = dataframe.loc[day, reference_patrol_post_name]

# Sialang has all rainfall and relhum values
fill_weather_dataframe_missing_values_with_reference_patrol_post(df_rain, reference_patrol_post_name='Sialang')
fill_weather_dataframe_missing_values_with_reference_patrol_post(df_relhum, reference_patrol_post_name='Sialang')

# %% Compute ET
jday_to_month = pd.to_datetime(df_relhum.Date).dt.month
  
# Compute air pressure in the station given sea level air pressure
air_pressure = evapotranspiration_fao.pressure_from_altitude(8.0) 

df_ET = pd.DataFrame()
for patrol_name in patrol_post_names:
    daily_ET = [0]*365
    
    for day in range(0, 365): # it was a leap year!
        n_month = jday_to_month[day]
        Tmax = df_maxtemp.loc[n_month-1, patrol_name]
        Tmin = df_mintemp.loc[n_month-1, patrol_name]
        Tmean = df_avgtemp.loc[n_month-1, patrol_name]
        relhum = df_relhum.loc[day, patrol_name]
        windspeed = df_windspeed.loc[day, 'Average Windspeed (m/s)']
        
        ET, _, _, _, _ = np.array(estimate_daily_et.compute_ET(day, Tmax=Tmax, Tave=Tmean, Tmin=Tmin,
                                                               RH=relhum, air_pressure=air_pressure, U=windspeed,
                                                               Kc=0.8, KRS=0.16))
        daily_ET[day] = ET

    df_ET[patrol_name] = np.array(daily_ET) * 0.001 # m/day

# %% Source = P - ET
df_source = df_rain - df_ET

#%% Read WTD transect dipwell data
df_wtd_transects = pd.read_csv(Path.joinpath(data_parent_folder, 'Raw csv/WTL_PatPos_2021.csv'), sep=';')
df_wtd_transects_canal = pd.read_csv(Path.joinpath(data_parent_folder, 'Raw csv/WaterLevel_Adjusted_2021.csv'), sep=';')

# Change units cm -> m
df_wtd_transects.loc[:, df_wtd_transects.columns != 'Date'] = df_wtd_transects.loc[:, df_wtd_transects.columns != 'Date'] * 0.01
df_wtd_transects_canal.loc[:, df_wtd_transects_canal.columns != 'Date'] = df_wtd_transects_canal.loc[:, df_wtd_transects_canal.columns != 'Date'] * 0.01

# Make dict of dataframes separated by transect
dict_of_transects = {}
relevant_transects = ['SI', 'SE', 'RA'] # Remove MC and LE because gradient might not be parallel to transect
for small_name in relevant_transects:
    df_wtd_transects[small_name + '.0'] = df_wtd_transects_canal[small_name]
    all_col_names_with_name = [name for name in df_wtd_transects.columns if small_name in name]
    dict_of_transects[small_name] = df_wtd_transects[all_col_names_with_name]

# %% Transect sensor info
# Elevation Data from DIpwell Elevation Report .xslx
dipwell_separation = [0, 5, 25, 50, 100, 150] # metres
dipwell_heights = {'MC': [0, 0.8, 1.21, 1.01, 1.09, 0.93],
                  'SI': [0, 1.19, 0.9, 1.19, 1.1, 1.07],
                  'RA': [0, 1.1, 1.16, 1.04, 1.03, 1.45],
                  'SE': [0, 0.76, 1.07, 0.93, 1.21, 1.21],
                  'LE': [0, 0.67, 0.58, 0.76, 1.06, 1.11]}

MESH = np.linspace(start=dipwell_separation[0], stop=dipwell_separation[-1], num=NX)

EXTRA_PEAT_DEPTH = 8.0 # Shifts dem upwards, i.e., changes reference datum to make sure water doesn't go below zero
transect_dem = {}
for name, heights in dipwell_heights.items():
    inter_func = interp1d(x=dipwell_separation, y=heights)
    transect_dem[name] = inter_func(MESH) + EXTRA_PEAT_DEPTH

#%% Choose data
# Uncomment to print bad days.

print('Bad (NaN and outliers) days per transect: ')
bad_data_per_transect = {}
for name, df in dict_of_transects.items():
    # outliers
    zscore_more_than_3std = (df.dropna().apply(zscore) > 3).all(axis=1)
    
    # Nans
    row_has_NaN = df.isnull().any(axis=1)
    
    # Append those values to list
    bad_data_per_transect[name] = list(row_has_NaN.index[row_has_NaN]) + list(zscore_more_than_3std.index[zscore_more_than_3std])
    
    print(name, bad_data_per_transect[name])

jday_bounds = {'SI':[50, 100],
               'SE':[150, 200],
               'RA':[300, 350]
               }

# %% Create initial zeta 
initial_zetas = {}
for name in relevant_transects:
    df_dipwell = dict_of_transects[name]
    used_dipwells = [name + '.0', name + '.1', name + '.2', name + '.3', name + '.4', name + '.5']
    ini_z = [df_dipwell.iloc[0][sn] for sn in used_dipwells]
    inter_func = interp1d(x=dipwell_separation, y=ini_z)
    initial_zetas[name] = inter_func(MESH)
    
#%% Make metadictionary of all the relevant data

def slice_by_julian_days(all_data_by_transect, jday_bounds):
    dfs_sliced_relevant_transects = {}
    for key, df in all_data_by_transect.items():
        sliced_df = df.loc[jday_bounds[key][0]:jday_bounds[key][1]]
        dfs_sliced_relevant_transects[key] = sliced_df
    
    return dfs_sliced_relevant_transects

data_dict = {}
for name in relevant_transects:
    df = dict_of_transects[name].copy()
    df['source'] = df_source[abbreviation_to_name[name]]
    # slice data
    df = df.loc[jday_bounds[name][0] : jday_bounds[name][1]]
    
    # append
    data_dict[name] = df

def get_sensor_measurements(df, t_name):
    zeta_left_diris = np.array(df[t_name + '.0'])
    zeta_right_diris = np.array(df[t_name + '.5'])
    zeta_middle_measurements = df[[t_name + '.1', t_name + '.2', t_name + '.3', t_name + '.4']].to_numpy()
    
    return zeta_left_diris, zeta_middle_measurements, zeta_right_diris


# %% EMCEE stuff

def log_prior(params):
    s1, s2, t1, t2 = params
    # uniform priors everywhere.
    if 0<s1<1000 and -10<s2<10 and 0<t1<1000 and 0<t2<100: 
        return 0.0
    return -np.inf        
 
def log_likelihood(params):
    log_like = 0 # initialize returned quantity
    s1, s2, t1, t2 = params

    for name in relevant_transects:
        dem = transect_dem[name]
        df = data_dict[name]
                
        ZETA_LEFT_DIRIS, zeta_middle_measurements, ZETA_RIGHT_DIRIS = get_sensor_measurements(df=df, t_name=name)
        source = df['source'].to_numpy()
        
        NDAYS = len(ZETA_LEFT_DIRIS) - 1 # First datapoint is for initial condition

        zeta_ini = initial_zetas[name]
        
        h_ini = h_from_zeta(zeta_ini, dem)
        theta = theta_from_h(h_ini, dem, s1, s2)
        
        try:
            #Transient hydro
            # previous day's info (BC at canal and last sensor)
            # are used to compute next day's wtd in the middle, so there is no data to test
            # the prediction for the middle wtd coming from the last boundary conditions.
            SOURCE = source[:-1]
            ZETA_LEFT_DIRIS = ZETA_LEFT_DIRIS[:-1]
            ZETA_RIGHT_DIRIS = ZETA_RIGHT_DIRIS[:-1]

            solution_zeta = solve_transient_both_diri(theta=theta,
                                                dx=dx, dt=dt, NDAYS=NDAYS, N=NX,
                                                dem=dem, b=b, s1=s1, s2=s2, t1=t1, t2=t2, SOURCE=SOURCE,
                                                ZETA_LEFT_DIRIS=ZETA_LEFT_DIRIS, ZETA_RIGHT_DIRIS=ZETA_RIGHT_DIRIS,
                                                MAX_INTERNAL_NITER=MAX_INTERNAL_NITER, WEIGHT=WEIGHT,
                                                REL_TOL=REL_TOL, ABS_TOL=ABS_TOL, verbose=False)
        
        except Exception as e:
            print(e)
            return -np.inf 
        
        # Compare
        # First, interpolate predicted value at the middle sensor location
        zeta_sol_middle_sensors_all_days = np.zeros(shape=(NDAYS, zeta_middle_measurements.shape[1]))
        for day in range(NDAYS):
            zeta_sol_interp_func = interp1d(x=MESH, y=solution_zeta[day])
            zeta_sol_middle_sensors = zeta_sol_interp_func(dipwell_separation[1:-1]) # first and last sensors are BC
            zeta_sol_middle_sensors_all_days[day] = zeta_sol_middle_sensors
        
        sigma2 = SENSOR_MEASUREMENT_ERR ** 2
        log_like += -0.5 * np.sum((zeta_middle_measurements[1:] - zeta_sol_middle_sensors_all_days) ** 2 / sigma2 + np.log(sigma2))
    
    return log_like

def log_probability(params):
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood(params)
    if not np.isfinite(ll).any(): # if Error in hydro computation
        return -np.inf
    return lp + ll


def gen_positions_for_walkers(n_walkers, n_params):
    # Generate based on true values + noise. TODO: change in the future!
    ini_values =  [0.1, -0.4, 10, 2] # s1, s2, t1, t2
    true_values = np.array([ini_values,]*n_walkers)
    noise = (np.random.rand(n_walkers, n_params) -0.5) # random numbers in (-0.5, +0.5)
    return true_values + noise
 
if N_CPU > 1:  
    with Pool(N_CPU) as pool:
            
        pos = gen_positions_for_walkers(N_WALKERS, N_PARAMS)
       
        nwalkers, ndim = pos.shape
         
        # save chain to HDF5 file
        fname = output_parent_folder.joinpath("mcmc_result_chain_FC_NEW.h5")
        backend = emcee.backends.HDFBackend(fname)
        backend.reset(nwalkers, ndim) # commenting this line: continue from stored markovchain
        
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, pool=pool,
                                        backend=backend,
                                        moves=[(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2),])
        sampler.run_mcmc(pos, MCMC_STEPS, progress=True)
         
elif N_CPU == 1: # single processor
    pos = gen_positions_for_walkers(N_WALKERS, N_PARAMS)
    nwalkers, ndim = pos.shape
     
    # save chain to HDF5 file
    fname = output_parent_folder.joinpath("mcmc_result_chain_FC.h5")
    backend = emcee.backends.HDFBackend(fname)
    backend.reset(nwalkers, ndim) # commenting this line: continue from stored markovchain
     
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, backend=backend,
                                    moves=[(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2),])
    sampler.run_mcmc(pos, MCMC_STEPS, progress=True)




# %%
