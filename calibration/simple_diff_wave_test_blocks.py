#%%
import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
import networkx as nx
import numpy as np
import pickle

import sys
sys.path.insert(1, r'C:\Users\03125327\github\blopti_dev')
from classes.channel_network import ChannelNetwork
from classes.channel_hydrology import set_up_channel_hydrology, CWLHydroParameters
from classes.channel_network import ChannelNetwork
import math_preissmann, math_diff_wave
import preprocess_data
import cwl_utilities

#%% Prepare graph
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
    BLOCK_POSITIONS = [int(N_NODES/2), 3*int(N_NODES/4)]
    graph = nx.from_numpy_array(cnm.T, create_using=nx.DiGraph)
    for n in range(n_nodes):
        graph.nodes[n]['DEM'] = dem[n]
        graph.nodes[n]['q'] = external_source_q
        if n in BLOCK_POSITIONS:
            graph.nodes[n]['is_block'] = True
        else:
            graph.nodes[n]['is_block'] = False
        
    return graph
#%% Prepare simus

# Parameters
N_NODES = 20
EXTERNAL_SOURCE_q = 0.0
CHANNEL_WIDTH = 5
INITIAL_WATER_HEIGHT = 20.0
DAM_DROP = 1 # drop of the dam (m)
NDAYS = 24*60
CANAL_LENGTH = 1000 # in m
dx = int(CANAL_LENGTH/N_NODES)
dt = 100
CHANNEL_BOTTOM_BELOW_DEM = 4 # m
BOTTOM_HEIGHT_DIFF = 1 # height difference between channel bed extremes (m)
CHANNEL_WIDTH = 4 # m
N_MANNING = 0.3

graph = prepare_graph(n_nodes=N_NODES, bottom_height_diff=BOTTOM_HEIGHT_DIFF, external_source_q=EXTERNAL_SOURCE_q, initial_water_height=INITIAL_WATER_HEIGHT)
graph = graph.reverse(copy=True)

# Plot graph
# options = {
#         "font_size": 10,
#         "node_size": 500,
#         "node_color": "white",
#         "edgecolors": "black",
#         "linewidths": 1,
#         "width": 1,
#         "with_labels":True}
# nx.draw_planar(graph, **options)

channel_network = ChannelNetwork(
    graph, block_height_from_surface=0.1, block_coeff_k=2000,
    y_ini_below_DEM=-0.2, Q_ini_value=1.0, channel_bottom_below_DEM=CHANNEL_BOTTOM_BELOW_DEM,
    y_BC_below_DEM=0.0, Q_BC=0.0, channel_width=CHANNEL_WIDTH, work_without_blocks=True, is_components=True)

cwl_params = CWLHydroParameters(g=9.8, dt=dt, dx=dx, n1=5, n2=1,
                                porous_threshold_below_dem = 3.0,
                                preissmann_a=0.6, rel_tol=1e-5, abs_tol=1e-5,
                                max_niter_newton=int(1e5), max_niter_inexact=int(1e3),
                                ntimesteps=1, ncpus=1,
                                weight_A=1e-1, weight_Q=1e-1)

cwl_hydro = set_up_channel_hydrology(model_type='diff-wave-implicit',
                                    cwl_params=cwl_params,
                                    cn=channel_network)

#%% Simulate
y_sol = math_diff_wave.simulate_one_component(general_params=cwl_params, channel_network=channel_network)
# %% Plot
plt.figure()
plt.bar(x=channel_network.block_nodes, height=channel_network.block_heights, width=0.8, color='black')
plt.plot(channel_network.y, label='initial')
plt.plot(y_sol, label='solution')
plt.plot(channel_network.dem, label='surface')
plt.ylim((channel_network.dem.min()-0.1, channel_network.dem.max()+0.3))

plt.legend()

# %%
