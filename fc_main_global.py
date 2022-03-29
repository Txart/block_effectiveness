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

import fc_get_data

if platform.system() == 'Linux': # Working on my own machine
    parent_directory = Path(r'~/Programming/GitHub/blopti_dev')
    
    sys.path.insert(1, r'~/Programming/GitHub/blopti_dev') # insert at 1, 0 is the script path
    mesh_fn = r"/home/txart/Dropbox/PhD/Computation/ForestCarbon/2021 SMPP WTD customer work/qgis_derivated_data/mesh/mesh/mesh_0.05.msh2" # triangular mesh

elif platform.system() == 'Windows':  
#    Imports from other folders
    sys.path.insert(1, r'C:\Users\03125327\github\blopti_dev') # insert at 1, 0 is the script path
    data_parent_folder = Path(r"C:\Users\03125327\Dropbox\PhD\Computation\ForestCarbon\2021 SMPP WTD customer work\0. Raw Data\Raw csv")

    parent_directory = Path(r"C:\Users\03125327\github\blopti_dev")
    # use unstructured mesh from gmsh file
    mesh_fn = r"C:\Users\03125327\Dropbox\PhD\Computation\ForestCarbon\2021 SMPP WTD customer work\qgis_derivated_data\mesh\mesh_0.05.msh2" # triangular mesh
    

    
from classes.channel_network import ChannelNetwork
from classes.channel_hydrology import set_up_channel_hydrology, CWLHydroParameters
from classes.peatland import Peatland
from classes.peatland_hydrology import PeatlandHydroParameters, set_up_peatland_hydrology
import preprocess_data


# %% Prepare data
#graph = cwl_preprocess_data.load_graph(load_from_pickled=True)
graph = pickle.load(open(parent_directory.joinpath("ForestCarbon2022/canal_network_matrix_50meters.p"), "rb"))
#graph = graph.reverse(copy=True)

fn_pointers = parent_directory.joinpath(r'ForestCarbon2022/file_pointers.xlsx')
filenames_df = pd.read_excel(fn_pointers, header=2, dtype=str)
fn_dem = Path(filenames_df[filenames_df.Content == 'DEM'].Path.values[0])
dem = preprocess_data.read_raster(fn_dem)

channel_network = ChannelNetwork(
    graph, block_height_from_surface=0.5, block_coeff_k=2000.0,
    y_ini_below_DEM=0.0, Q_ini_value=0.0, channel_bottom_below_DEM=8.0,
    y_BC_below_DEM=0.0, Q_BC=0.1, channel_width=3.5, work_without_blocks=True)

peatland = Peatland(
    cn=channel_network, fn_pointers=parent_directory.joinpath(r'ForestCarbon2022/file_pointers.xlsx'))

peat_hydro_params = PeatlandHydroParameters(
    dt=1/24, # dt in days
    dx=50, # dx in meters, only used if structured mesh
    nx=dem.shape[0], ny=dem.shape[1], max_sweeps=1000, fipy_desired_residual=1e-4,
    s1=0.884, s2=0.85, t1=16.61, t2=2.3,
    use_several_weather_stations=True)

# Mesh has some cell centers outside of the dem. This is a quickfix.
peat_hydro_params.bigger_dem_raster_fn = r"C:\Users\03125327\Dropbox\PhD\Computation\ForestCarbon\2021 SMPP WTD customer work\qgis_derivated_data\dtm_big_area_depth_padded.tif"


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
                                rel_tol=1e-3, abs_tol=1e-3, weight_A=5e-2, weight_Q=5e-2, ncpus=1,
                                downstream_diri_BC=False)

cwl_hydro = set_up_channel_hydrology(model_type='diff-wave-implicit',
                                        cwl_params=cwl_params,
                                        cn=channel_network)

hydro = set_up_peatland_hydrology(mesh_fn=mesh_fn, model_coupling='darcy',
                                  peatland=peatland, peat_hydro_params=peat_hydro_params,
                                  channel_network=channel_network, cwl_params=cwl_params)

#%% Read weather data
df_p_minus_et = fc_get_data.get_P_minus_ET_dataframe(data_parent_folder)

#%% Simulation functions

def simulate_one_timestep(hydro, cwl_hydro):
    # results stored in hydro.zeta and other class attributes
    
    # "predictive" step: compute the CWL source term, q, using the peat hydro model
    theta = hydro.create_theta_from_zeta(hydro.zeta)

    theta_sol = hydro.run(mode='theta', fipy_var=theta)
    
    zeta_sol = hydro.zeta_from_theta(theta_sol)
    y_prediction_at_canals = hydro.convert_zeta_to_y_at_canals(zeta_sol)
    y_prediction_array = hydro.cn.from_nodedict_to_nparray(y_prediction_at_canals)
    q = cwl_hydro.predict_q_for_next_timestep(y_prediction_at_canals=y_prediction_array,
                                              y_at_canals=hydro.cn.y)
    # Add q to each component graph
    q_nodedict = hydro.cn.from_nparray_to_nodedict(q)
    for component_cn in cwl_hydro.component_channel_networks:
        component_cn.q = component_cn.from_nodedict_to_nparray(q_nodedict)
    
    # CWL solution step
    # cwl_hydro.run() takes y, q and others from each component's attribute variables
    cwl_hydro.run() # solution stored in cwl_hydro.y_solution (or Q_solution also if Preissmann)
    y_solution_at_canals = cwl_hydro.y_solution # y is water height above common ref point
    
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
      
    # Simulate WTD
    hydro.theta = hydro.create_theta_from_zeta(hydro.zeta) 
    hydro.theta = hydro.run(mode='theta', fipy_var=hydro.theta)  # solution is saved into class attribute
    hydro.zeta = hydro.create_zeta_from_theta(hydro.theta)
    
    return hydro, cwl_hydro

# %% Some computations
# Complete saturation
hydro.zeta = hydro.create_uniform_fipy_var(uniform_value=0.0, var_name='zeta')
MEAN_P_MINUS_ET = -0.004 # m/day
hydro.verbose = True

# Pre-initial thing for smoothing beginning. Simulate some days of only WTD drawdown, no CWL
for day in range(0):
    print(day)
    hydro.set_sourcesink_variable(value=MEAN_P_MINUS_ET) # m/hour
    hydro.ph_params.dt = 1 # dt in days
    hydro.theta = hydro.create_theta_from_zeta(hydro.zeta) 
    hydro.theta = hydro.run(mode='theta', fipy_var=hydro.theta)  # solution is saved into class attribute
    hydro.zeta = hydro.create_zeta_from_theta(hydro.theta)

# Set initial zeta from best pickled file
best_ic_fn = "C:/Users/03125327/github/blopti_dev/ForestCarbon2022/2d_calibration_output/params_number_9/best_initial_zeta.p"
zetavalue = pickle.load(open(best_ic_fn, 'rb'))
hydro.zeta = fp.CellVariable(name='zeta', mesh=hydro.mesh, value=zetavalue, hasOld=True)

# Blocks?
hydro.cn.work_without_blocks = True

# Params
cwl_hydro.cwl_params.porous_threshold_below_dem = 1.5
cwl_hydro.cwl_params.n1 = 3
cwl_hydro.cwl_params.n2 = 1
hydro.ph_params.s1 = 0.80795613
hydro.ph_params.s2 = 0.40543195
hydro.ph_params.t1 = 900
hydro.ph_params.t2 = 5


N_DAYS = 365
for day in tqdm(range(N_DAYS), position=0, desc='DAYS'):
    
    # P-ET sink/source term at each weather station
    hydro.ph_params.use_several_weather_stations = True
    sources_dict = df_p_minus_et.loc[day].to_dict() # sources_dict = {'Ramin':1.0, 'Serindit':0.002, 'Buring':0.5, 'Lestari':-0.004, 'Sialang':1, 'MainCamp':2}
    hydro.set_sourcesink_variable(value=sources_dict)

    zeta_time0 = hydro.zeta.value

    hydro.ph_params.dt = 1/24 # dt in days
    hydro.cn_params.dt = 3600 # dt in seconds
    
    for hour in tqdm(range(24), position=1): # hourly timestep
        zeta_hour0 = hydro.zeta.value
        hydro, cwl_hydro = simulate_one_timestep(hydro, cwl_hydro)
        print(f'>>>>> HOURLY at hour {hour} norm(diff(zeta)) = {np.linalg.norm(zeta_hour0 - hydro.zeta.value)} ')
    
    zeta_time1 = hydro.zeta.value
    print('norm(diff(zeta)) = ', np.linalg.norm(zeta_time0 - zeta_time1))

    hydro.save_fipy_var_in_raster_file(fipy_var=hydro.zeta,
                                   out_raster_fn=r'C:/Users/03125327/github/blopti_dev/ForestCarbon2022/2d_calibration_output/params_number_9_NODAMS/zeta_after_' + str(day) + '_DAYS.tif',
                                   interpolate=True)


#%% 
# %%
