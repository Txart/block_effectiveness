# %%
import numpy as np
from pathlib import Path
import networkx as nx
import copy
from tqdm import tqdm
import fipy as fp
import pickle
import platform
import pandas as pd
from pathlib import Path
import sys
from scipy.spatial import distance
import multiprocessing as mp
import argparse

import get_data
import preprocess_data
from classes.channel_network import ChannelNetwork
from classes.channel_hydrology import set_up_channel_hydrology, CWLHydroParameters
from classes.peatland import Peatland
from classes.peatland_hydrology import PeatlandHydroParameters, set_up_peatland_hydrology
from classes.parameterizations import ExponentialBelowOneAboveStorageWithDepth

if platform.system() == 'Linux': # Working on my own machine
    parent_directory = Path(r'/users/urzainqu/paper2')
    data_parent_folder = parent_directory.joinpath('data/Raw csv')
    fn_pointers = parent_directory.joinpath(r'file_pointers_csc.xlsx')
    
    parser = argparse.ArgumentParser(description='Run 2d calibration')
    
    parser.add_argument('--ncpu', default=1, help='(int) Number of processors', type=int)

    args = parser.parse_args()
    
    N_CPU = args.ncpu    

elif platform.system() == 'Windows':  
    parent_directory = Path(r"C:\Users\03125327\github\paper2")
    data_parent_folder = Path(r"data\Raw csv")
    fn_pointers = parent_directory.joinpath(r'file_pointers.xlsx')
    N_CPU = 1
    
#%% Read weather data
df_p_minus_et = get_data.get_P_minus_ET_dataframe(data_parent_folder)

# %% Prepare data
filenames_df = pd.read_excel(fn_pointers, header=2, dtype=str, engine='openpyxl')
fn_dem = Path(filenames_df[filenames_df.Content == 'DEM'].Path.values[0])
dem = preprocess_data.read_raster(fn_dem)
mesh_fn = Path(filenames_df[filenames_df.Content == 'mesh'].Path.values[0])

graph_fn = Path(filenames_df[filenames_df.Content == 'channel_network_graph_pickle'].Path.values[0])
graph = pickle.load(open((graph_fn), "rb"))

channel_network = ChannelNetwork(
    graph, block_height_from_surface=0.0, block_coeff_k=2000.0,
    y_ini_below_DEM=-0.0, Q_ini_value=0.0, channel_bottom_below_DEM=8.0,
    y_BC_below_DEM=-0.0, Q_BC=0.0, channel_width=3.5, work_without_blocks=False)

peatland = Peatland(cn=channel_network, fn_pointers=fn_pointers)

peat_hydro_params = PeatlandHydroParameters(
    dt=1/24, # dt in days
    dx=50, # dx in meters, only used if structured mesh
    nx=dem.shape[0], ny=dem.shape[1], max_sweeps=1000, fipy_desired_residual=1e-5,
    s1= 0.0, s2=0.0, t1=0, t2=0,
    use_several_weather_stations=True)

# Set up cwl computation
cwl_params = CWLHydroParameters(g=9.8, # g in meters per second
                                dt=3600, # dt in seconds
                                dx=50, # dx in meters
                                ntimesteps=1, # outer loop. Number of steps simulated
                                preissmann_a=0.6,
                                porous_threshold_below_dem = 3.0, # threshold position for n_manning in metres below dem
                                n1=5, # params for n_manning
                                n2=1,
                                max_niter_newton=int(1e5), max_niter_inexact=int(50),
                                rel_tol=1e-5, abs_tol=1e-5, weight_A=5e-2, weight_Q=5e-2, ncpus=1,
                                downstream_diri_BC=0.0)

cwl_hydro = set_up_channel_hydrology(model_type='diff-wave-implicit-inexact',
                                        cwl_params=cwl_params,
                                        cn=channel_network)

parameterization = ExponentialBelowOneAboveStorageWithDepth(peat_hydro_params)

hydro = set_up_peatland_hydrology(mesh_fn=mesh_fn, model_coupling='darcy',
                                  use_scaled_pde=False, zeta_diri_bc=-0.2,
                                  force_ponding_storage_equal_one=False,
                                  peatland=peatland, peat_hydro_params=peat_hydro_params,
                                  parameterization=parameterization,
                                  channel_network=channel_network, cwl_params=cwl_params)


#%% Simulation functions

def simulate_one_timestep_simple_two_step(hydro, cwl_hydro):
    # Alternative to simulate_one_timestep that only uses 1 fipy solution per timestep
    zeta_before = hydro.zeta
    # Simulate WTD
    hydro.theta = hydro.create_theta_from_zeta(hydro.zeta) 
    
    hydro.theta = hydro.run(mode='theta', fipy_var=hydro.theta)  # solution is saved into class attribute
    hydro.zeta = hydro.create_zeta_from_theta(hydro.theta)
    
    new_y_at_canals = hydro.convert_zeta_to_y_at_canals(hydro.zeta)
    y_prediction_array = hydro.cn.from_nodedict_to_nparray(new_y_at_canals)
    theta_prediction_array = hydro.parameterization.theta_from_zeta(zeta=y_prediction_array - hydro.cn.dem, dem=hydro.cn.dem)
    theta_previous_array = hydro.parameterization.theta_from_zeta(zeta=hydro.cn.y - hydro.cn.dem, dem=hydro.cn.dem)
    theta_diff = theta_prediction_array - theta_previous_array
    q = cwl_hydro.predict_q_for_next_timestep(theta_difference=theta_diff, seconds_per_timestep=hydro.cn_params.dt)
    
    # Add q to each component graph
    q_nodedict = hydro.cn.from_nparray_to_nodedict(q)
    for component_cn in cwl_hydro.component_channel_networks:
        component_cn.q = component_cn.from_nodedict_to_nparray(q_nodedict)
    
    # CWL solution step
    # cwl_hydro.run() takes y, q and others from each component's attribute variables
    cwl_hydro.run() # solution stored in cwl_hydro.y_solution (or Q_solution also if Preissmann)
    y_solution_at_canals = cwl_hydro.y_solution # y is water height above common ref point
    
    # Distribute canal ponding water throughout the cell
    zeta_at_canals = cwl_hydro.convert_y_nodedict_to_zeta_at_canals(y_solution_at_canals)
    ponding_canal_nodes = cwl_hydro._find_zeta_in_ponding_canal_nodes(y_solution_at_canals)
    zeta_at_canals = hydro.distribute_canal_ponding_water_throughout_cell(zeta_at_canals, ponding_canal_nodes)
    y_solution_at_canals = cwl_hydro.convert_zeta_nodedict_to_y_at_canals(zeta_at_canals)
    
    # Append solution value at canals to each graph in the components
    for component_cn in cwl_hydro.component_channel_networks:
        component_cn.y = component_cn.from_nodedict_to_nparray(y_solution_at_canals)

    # Here I append the computed cwl to the global channel_network, both as channel_network.y,
    #  and also to each of the components in cwl_hydro.component_channel_networks
    # The components part is needed for the cwl computation, and the global graph part
    # is needed to taake into account nodes in the channel network outside the components
    zeta_at_canals = cwl_hydro.convert_y_nodedict_to_zeta_at_canals(y_solution_at_canals)
    mean_zeta_value = np.mean(list(zeta_at_canals.values()))
    for n in cwl_hydro.nodes_not_in_components:
        y_solution_at_canals[n] = mean_zeta_value + cwl_hydro.cn.graph.nodes[n]['DEM']
    y_sol_all_canals = hydro.cn.from_nodedict_to_nparray(y_solution_at_canals)
    hydro.cn.y = y_sol_all_canals
    
    # Set solution of CWL simulation into fipy variable zeta for WTD simulation
    zeta_at_canals = cwl_hydro.convert_y_nodedict_to_zeta_at_canals(y_solution_at_canals)
    zeta_array = hydro.cn.from_nodedict_to_nparray(zeta_at_canals)
    hydro.set_canal_values_in_fipy_var(channel_network_var=zeta_array, fipy_var=hydro.zeta)
      
    
    
    return hydro, cwl_hydro

#%% Initial condition

def find_best_initial_condition(param_number, PARAMS, hydro, cwl_hydro, parent_directory, output_folder_path):
    hydro.ph_params.s1 = PARAMS.loc[param_number, 's1']
    hydro.ph_params.s2 = PARAMS.loc[param_number, 's2']
    hydro.ph_params.t1 = PARAMS.loc[param_number, 't1']
    hydro.ph_params.t2 = PARAMS.loc[param_number, 't2']
    cwl_hydro.cwl_params.porous_threshold_below_dem = PARAMS.loc[param_number, 'porous_threshold']
    cwl_hydro.cwl_params.n1 = PARAMS.loc[param_number, 'n1']
    cwl_hydro.cwl_params.n2 = PARAMS.loc[param_number, 'n2']
    
    hydro.parameterization = ExponentialBelowOneAboveStorageWithDepth(hydro.ph_params)
    
    # Begin from complete saturation
    hydro.zeta = hydro.create_uniform_fipy_var(uniform_value=-0.0, var_name='zeta')
    
    # Initialize returned variable
    best_initial_zeta = hydro.zeta.value
    best_fitness = np.inf
    
    # Read day 0 sensor coords and values
    # The pickle file was produced elsewhere, probably in scratch.py
    fn_sensor_pickle = parent_directory.joinpath("initial_sensor_pickle.p")
    sensor_coords, sensor_measurements = pickle.load(open(fn_sensor_pickle, 'rb'))
    
    # Get mesh cell center posiitions to compare to sensor values
    mesh_cell_centers = hydro.mesh.cellCenters.value.T
    
    MEAN_P_MINUS_ET = -0.0075 # m/day. SOmething like 2,5x the daily ET to speed things up.
    hydro.ph_params.use_several_weather_stations = False
    hydro.set_sourcesink_variable(value=MEAN_P_MINUS_ET)

    N_DAYS = 100
    day = 0
    needs_smaller_timestep = True # If True, start day0 with a small timestep to smooth things
    NORMAL_TIMESTEP = 24 # Hourly
    SMALLER_TIMESTEP = 100
    while day < N_DAYS:
        # Variables from current timestep for flexible solve
        hydro_old = copy.deepcopy(hydro)
        cwl_hydro_old = copy.deepcopy(cwl_hydro)
        if not needs_smaller_timestep:
            internal_timesteps = NORMAL_TIMESTEP
        elif needs_smaller_timestep:
            internal_timesteps = SMALLER_TIMESTEP 
            
        hydro.ph_params.dt = 1/internal_timesteps # dt in days
        hydro.cn_params.dt = 86400/internal_timesteps # dt in seconds
        # Add pan ET
        hydro.set_sourcesink_variable(value=MEAN_P_MINUS_ET) # we need to insert this here, otherwise the panET is added every day.
        hydro.sourcesink = hydro.sourcesink - hydro.compute_pan_ET_from_ponding_water(hydro.zeta)
        
        try:
            solution_function = simulate_one_timestep_simple_two_step
            
            for i in tqdm(range(internal_timesteps)): # internal timestep
                hydro, cwl_hydro = solution_function(hydro, cwl_hydro)

                
        except Exception as e:
            if internal_timesteps == NORMAL_TIMESTEP:
                print(f'Exception in INITIAL RASTER computation of param number {param_number} at day {day}: ', e,
                      ' I will retry with a smaller timestep')
                needs_smaller_timestep = True
                hydro = hydro_old
                cwl_hydro = cwl_hydro_old
                continue
        
            elif internal_timesteps == SMALLER_TIMESTEP:
                print(f"Another exception caught in INITIAL RASTER computation with smaller timestep: {e}. ABORTING")
                break
        
        else: # try was successful
            # Compute fitness
            dists = distance.cdist(sensor_coords, mesh_cell_centers)
            mesh_number_corresponding_to_sensor_coords = np.argmin(dists, axis=1)
            zeta_values_at_sensor_locations = np.array([hydro.zeta.value[m] for m in mesh_number_corresponding_to_sensor_coords])
            current_fitness = np.linalg.norm(sensor_measurements - zeta_values_at_sensor_locations)

            # If fitness is improved, update initial zeta. And pickle it for future use
            is_fitness_improved = current_fitness < best_fitness
            if is_fitness_improved:
                print(f"zeta at day {day} is better than the previous best, with a fitness difference of {best_fitness - current_fitness} points")
                best_initial_zeta = hydro.zeta.value
                best_fitness = current_fitness
            
                fn_pickle = output_folder_path.joinpath('best_initial_zeta.p')

                pickle.dump(best_initial_zeta, open(fn_pickle, 'wb'))
                
            # go to next day
            day = day + 1
            needs_smaller_timestep = False
                
    return best_initial_zeta


#%% main function

def run_daily_computations(hydro, cwl_hydro, df_p_minus_et, internal_timesteps, day):
    
    hydro.ph_params.dt = 1/internal_timesteps # dt in days
    cwl_hydro.cwl_params.dt = 86400/internal_timesteps # dt in seconds
    
    # # P-ET sink/source term at each weather station
    hydro.ph_params.use_several_weather_stations = True
    sources_dict = df_p_minus_et.loc[day].to_dict() # sources_dict = {'Ramin':1.0, 'Serindit':0.002, 'Buring':0.5, 'Lestari':-0.004, 'Sialang':1, 'MainCamp':2}
    hydro.set_sourcesink_variable(value=sources_dict)
    hydro.sourcesink = hydro.sourcesink - hydro.compute_pan_ET_from_ponding_water(hydro.zeta) # Add pan ET
    
    zeta_t0 = hydro.zeta.value

    solution_function = simulate_one_timestep_simple_two_step
        
    for hour in range(internal_timesteps): # hourly timestep
        hydro, cwl_hydro = solution_function(hydro, cwl_hydro)

    zeta_t1 = hydro.zeta.value
    
    return hydro, cwl_hydro


def write_output_zeta_raster(zeta, full_folder_path, day):
    out_raster_fn = Path.joinpath(full_folder_path, f"zeta_after_{day}_DAYS.tif")
    hydro.save_fipy_var_in_raster_file(fipy_var=zeta, out_raster_fn=out_raster_fn, interpolation='linear')
    
    return None


def produce_family_of_rasters(param_number, PARAMS, hydro, cwl_hydro, df_p_minus_et, parent_directory):
    hydro.ph_params.s1 = PARAMS.loc[param_number, 's1']
    hydro.ph_params.s2 = PARAMS.loc[param_number, 's2']
    hydro.ph_params.t1 = PARAMS.loc[param_number, 't1']
    hydro.ph_params.t2 = PARAMS.loc[param_number, 't2']
    cwl_hydro.cwl_params.porous_threshold_below_dem = PARAMS.loc[param_number, 'porous_threshold']
    cwl_hydro.cwl_params.n1 = PARAMS.loc[param_number, 'n1']
    cwl_hydro.cwl_params.n2 = PARAMS.loc[param_number, 'n2']
    
    hydro.parameterization = ExponentialBelowOneAboveStorageWithDepth(hydro.ph_params)

    # Outputs will go here
    output_directory = Path.joinpath(parent_directory, 'output')
    out_rasters_folder_name = f"params_number_{param_number}"
    full_folder_path = Path.joinpath(output_directory, out_rasters_folder_name)
    

    best_initial_zeta_value = find_best_initial_condition(param_number, PARAMS,
                                                        hydro, cwl_hydro,
                                                        parent_directory, full_folder_path)

    hydro.zeta = fp.CellVariable(name='zeta', mesh=hydro.mesh, value=best_initial_zeta_value, hasOld=True)
    
    N_DAYS = 0
    day = 0
    needs_smaller_timestep = False
    NORMAL_TIMESTEP = 24 # Hourly
    SMALLER_TIMESTEP = 1000

    while day < N_DAYS:
        print(f'\n computing day {day}')
        
        if not needs_smaller_timestep:
            internal_timesteps = NORMAL_TIMESTEP
        elif needs_smaller_timestep:
            internal_timesteps = SMALLER_TIMESTEP
        
        hydro_test = copy.deepcopy(hydro)
        cwl_hydro_test = copy.deepcopy(cwl_hydro)    
        
        try:    
            hydro_test, cwl_hydro_test = run_daily_computations(hydro_test, cwl_hydro_test, df_p_minus_et, internal_timesteps, day)
                    
        except Exception as e:
            if internal_timesteps == NORMAL_TIMESTEP:
                print(f'Exception in computation of param number {param_number} at day {day}: ', e,
                      ' I will retry with a smaller timestep')
                needs_smaller_timestep = True
                continue
        
            elif internal_timesteps == SMALLER_TIMESTEP:
                print(f"Another exception caught with smaller timestep: {e}. ABORTING")
                return 0
        
        else: # try was successful
            # Update variables
            hydro = copy.deepcopy(hydro_test)
            cwl_hydro = copy.deepcopy(cwl_hydro_test)
            
            day += 1
            needs_smaller_timestep = False
            
            # write zeta to file
            if channel_network.work_without_blocks:
                write_output_zeta_raster(hydro.zeta, full_folder_path.joinpath('no_blocks'), day)
            else:
                write_output_zeta_raster(hydro.zeta, full_folder_path.joinpath('yes_blocks'), day)
                
            continue
                
    return 0

# %% Params



hydro.ph_params.dt = 1/24 # dt in days
hydro.cn_params.dt = 3600 # dt in seconds

# Read params
params_fn = Path.joinpath(parent_directory, '2d_calibration_parameters.xlsx')
PARAMS = pd.read_excel(params_fn, engine='openpyxl')
N_PARAMS = N_CPU
#%% Run multiprocessing csc
if platform.system() == 'Linux':
    if N_PARAMS > 1:
        hydro.verbose = False
        param_numbers = range(0, N_PARAMS)
        multiprocessing_arguments = [(param_number, PARAMS, hydro, cwl_hydro, df_p_minus_et, parent_directory) for param_number in param_numbers]
        with mp.Pool(processes=N_CPU) as pool:
            pool.starmap(produce_family_of_rasters, multiprocessing_arguments)
    
    elif N_PARAMS == 1:
        hydro.verbose = False
        param_numbers = [9]
        arguments = [(param_number, PARAMS, hydro, cwl_hydro, df_p_minus_et, parent_directory) for param_number in param_numbers]
        for args in arguments:
            produce_family_of_rasters(*args)    
    


#%% Run Windows
if platform.system() == 'Windows':
    hydro.verbose = True
    N_PARAMS = 1
    param_numbers = [9]
    arguments = [(param_number, PARAMS, hydro, cwl_hydro, df_p_minus_et, parent_directory) for param_number in param_numbers]

    for args in arguments:
        produce_family_of_rasters(*args)


# %%

# %%
