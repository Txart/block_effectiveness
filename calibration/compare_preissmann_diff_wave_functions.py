
#%%
import pickle
import sys
import networkx as nx
import numpy as np
import pickle

sys.path.insert(1, r'C:\Users\03125327\github\blopti_dev')
from classes.channel_hydrology import set_up_channel_hydrology, CWLHydroParameters
from classes.channel_network import ChannelNetwork
import math_preissmann, math_diff_wave
import preprocess_data
import cwl_utilities


# %% Prepare graph

def prepare_graph(n_nodes, bottom_height_diff, external_source_q, initial_water_height):
    REAL_DATA = False
    if REAL_DATA:
        #graph = cwl_preprocess_data.load_graph(load_from_pickled=True)
        graph = pickle.load(open("canal_network_matrix_with_q.p", "rb"))
        fn_output_dem_raster = r"C:\Users\03125327\github\blopti_dev\data\DTM_metres_clip.tif"
        dem = preprocess_data.read_raster(fn_output_dem_raster)

    graph_name = 'line'
    cnm = cwl_utilities.choose_graph(graph_name, n_nodes=n_nodes)
    dem = np.linspace(start=initial_water_height, stop=initial_water_height-bottom_height_diff,  num=n_nodes)

    # create graph
    graph = nx.from_numpy_array(cnm.T, create_using=nx.DiGraph)
    for n in range(n_nodes):
        graph.nodes[n]['DEM'] = dem[n]
        graph.nodes[n]['q'] = external_source_q
        
    return graph

#%%
def set_up_cwl_computation(graph, dx, channel_width, channel_bottom_below_dem, n_manning, model_type:str):    
    if model_type == 'diff-wave-implicit':
        dt = 60
        ntimesteps = 1
        
    elif model_type == 'preissmann':
        dt = 60
        ntimesteps = 1
        
    channel_network = ChannelNetwork(
        graph, block_nodes=[], block_heights_from_surface=[], block_coeff_k=2.0,
        y_ini_below_DEM=0.0, Q_ini_value=0.0, channel_bottom_below_DEM=channel_bottom_below_dem,
        y_BC_below_DEM=-1.0, Q_BC=0.0, channel_width=channel_width, set_node_neighbors=True)
    
    cwl_params = CWLHydroParameters(g=9.8, dt=dt, dx=dx, n_manning=n_manning,
                                    preissmann_a=0.6, rel_tol=1e-5, abs_tol=1e-5,
                                    max_niter_newton=int(1e5), max_niter_inexact=int(1e3),
                                    ntimesteps=ntimesteps, ncpus=1,
                                    weight_A=1e-1, weight_Q=1e-1)

    cwl_hydro = set_up_channel_hydrology(model_type=model_type,
                                        cwl_params=cwl_params,
                                        cn=channel_network)
    
    
    return channel_network, cwl_params, cwl_hydro


def run_simulations(mode, n_nodes, ndays, dx, initial_water_height, dam_drop,  bottom_height_diff, channel_bottom_below_dem, channel_width, n_manning, external_source_q):
    # mode can be 'gridsearch' or 'normal'. This changes returned quantities.
    graph = prepare_graph(n_nodes=n_nodes, bottom_height_diff=bottom_height_diff, external_source_q=external_source_q, initial_water_height=initial_water_height)
    
    channel_network, cwl_params, _ = set_up_cwl_computation(graph=graph, dx=dx,
                                                            channel_width=channel_width,
                                                            channel_bottom_below_dem=channel_bottom_below_dem,
                                                            n_manning=n_manning,
                                                            model_type='diff-wave-implicit')

    # Ini cond in equilibrium. By default, it is out of equilibrium
    channel_network.y = initial_water_height * np.ones(shape=n_nodes)
    channel_network.y[-1] = initial_water_height - dam_drop # BC

    df_y_dw = math_diff_wave.simulate_one_component_several_iter(ndays, channel_network, cwl_params)
    cwl_after_dw = df_y_dw.drop(labels=['DEM'], axis=1).to_numpy().T
    
    # Preissmann
    graph = prepare_graph(n_nodes=n_nodes, bottom_height_diff=bottom_height_diff, external_source_q=external_source_q, initial_water_height=initial_water_height)
    channel_network, cwl_params, _ = set_up_cwl_computation(graph=graph, dx=dx,
                                                            channel_width=channel_width,
                                                            channel_bottom_below_dem=channel_bottom_below_dem,
                                                            n_manning=n_manning,
                                                            model_type='preissmann')

    # Ini cond in equilibrium. By default, it is out of equilibrium
    channel_network.y = initial_water_height * np.ones(shape=n_nodes)
    channel_network.y[-1] = initial_water_height - dam_drop # BC

    try:
        df_y_pr, df_Q = math_preissmann.simulate_one_component_several_iter(ndays, channel_network, cwl_params)
        cwl_after_pr = df_y_pr.drop(labels=['DEM'], axis=1).to_numpy().T
        
        # Returned quantities
        # Differences between models
        diff_norms_at_timesteps = np.linalg.norm(cwl_after_dw - cwl_after_pr, axis=1)
        
        # Neglected quantities
        Q_pr = df_Q.to_numpy().T
        dQ_dt_norm = np.linalg.norm(np.gradient(Q_pr, axis=0), axis=1)
        Q2overA = Q_pr**2/cwl_after_pr
        dQ2overA_dx_norm  = np.linalg.norm(np.gradient(Q2overA, axis=1), axis=1)
        
        if mode == 'gridsearch':  # only interested in differences       
            return diff_norms_at_timesteps, dQ_dt_norm, dQ2overA_dx_norm
        elif mode == 'normal':  
            return cwl_after_dw, cwl_after_pr, Q_pr
        
    
    except:
        return None, None, None

