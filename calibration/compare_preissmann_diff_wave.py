

#%% Simulate

# Parameters
N_NODES = 50
EXTERNAL_SOURCE_q = 0.0
CHANNEL_WIDTH = 5
INITIAL_WATER_HEIGHT = 20.0
DAM_DROP = 1 # drop of the dam (m)
NDAYS = 24*60
CANAL_LENGTH = 1000 # in m
dx = int(CANAL_LENGTH/N_NODES)
CHANNEL_BOTTOM_BELOW_DEM = 4 # m
BOTTOM_HEIGHT_DIFF = 0 # height difference between channel bed extremes (m)
CHANNEL_WIDTH = 4 # m
N_MANNING = 0.3

cwl_after_dw, cwl_after_pr, Q_pr = run_simulations(mode='normal', n_nodes=N_NODES, ndays=NDAYS, dx=dx,
                                                                        initial_water_height=INITIAL_WATER_HEIGHT,
                                                                        dam_drop=DAM_DROP,
                                                                        bottom_height_diff=BOTTOM_HEIGHT_DIFF,
                                                                        channel_bottom_below_dem=CHANNEL_BOTTOM_BELOW_DEM,
                                                                        channel_width=CHANNEL_WIDTH,
                                                                        n_manning=N_MANNING,
                                                                        external_source_q=EXTERNAL_SOURCE_q)

diff_norms_at_timesteps = np.linalg.norm(cwl_after_dw - cwl_after_pr, axis=1)

# Neglected quantities
dQ_dt_norm = np.linalg.norm(np.gradient(Q_pr, axis=0), axis=1)
Q2overA = Q_pr**2/cwl_after_pr
dQ2overA_dx_norm  = np.linalg.norm(np.gradient(Q2overA, axis=1), axis=1)

 # diff_norm, dQ_dt_norm, dQ2overA_dx_norm = run_simulations(bottom_height_diff=b,
        #                                                         channel_bottom_below_dem=c,
        #                                                         n_manning=n)

            
        # diff_norms[niter] = diff_norm
        # dQ_dt_norms[niter] = dQ_dt_norm
        # dQ2overA_dx_norms[niter] = dQ2overA_dx_norm
        
        
# %% Plots

# In order to get channel parameters:
graph = prepare_graph(n_nodes=N_NODES, bottom_height_diff=BOTTOM_HEIGHT_DIFF,
                      external_source_q=EXTERNAL_SOURCE_q,
                      initial_water_height=INITIAL_WATER_HEIGHT)

channel_network = ChannelNetwork(
        graph, block_nodes=[], block_heights_from_surface=[], block_coeff_k=2.0,
        y_ini_below_DEM=0.0, Q_ini_value=0.0, channel_bottom_below_DEM=CHANNEL_BOTTOM_BELOW_DEM,
        y_BC_below_DEM=-1.0, Q_BC=0.0, channel_width=CHANNEL_WIDTH, set_node_neighbors=True)


x_real_distance = np.linspace(start=1, stop=CANAL_LENGTH, num=N_NODES)
plt.figure()
plt.title('norm(preissmann - diff_wave) at each timestep')
plt.xlabel('timestep (minutes)')
plt.plot(diff_norms_at_timesteps)

# Steady state equilibrium
plt.figure()
plt.title('Last snapshot')
plt.ylabel('canal water level (m.a.s.l.)')
plt.xlabel('canal reach (m)')
plt.ylim(np.min(channel_network.bottom), np.max(cwl_after_pr))
plt.plot(x_real_distance,  cwl_after_pr[-1], color='blue', label='preissmann')
plt.plot(x_real_distance, cwl_after_dw[-1], color='orange', label='diff_wave')
plt.fill_between(x_real_distance, channel_network.bottom, y2=10, color='brown', alpha=0.2)
plt.legend()


plt.figure()
plt.title('Neglected quantities')
plt.ylabel('$|quantity| (m^3/s^2)$')
plt.xlabel('timestep (minutes)')
plt.plot(dQ_dt_norm, color='green', label='$\partial Q/ \partial t$')
plt.plot(dQ2overA_dx_norm, color='purple', label='$\partial/\partial x (Q^2/A)$')
plt.legend()

# Froude number through time
# Fr = Q/sqrt(g*H)
froude_number = Q_pr/np.sqrt(9.8*(cwl_after_pr-channel_network.bottom))
max_froude_number = np.max(froude_number, axis=1)
min_froude_number = np.min(froude_number, axis=1)
mean_froude_number = np.mean(froude_number, axis=1)

plt.figure()
plt.title('Froude number.')
plt.ylabel('$F_r$')
plt.xlabel('timestep (minutes)')
plt.plot(mean_froude_number, color='blue', label='Mean $F_r$')
plt.plot(max_froude_number, color='red', label='Max $F_r$')
plt.plot(min_froude_number, color='green', label='Min $F_r$')
plt.legend()


# %% Animate
from matplotlib.animation import FuncAnimation 
# initializing a figure in 
# which the graph will be plotted
fig = plt.figure() 
# marking the x-axis and y-axis
axis = plt.axes(xlim =(0, CANAL_LENGTH), 
                ylim =(min(cwl_after_pr.min(),cwl_after_dw.min()),
                       max(cwl_after_pr.max(),cwl_after_dw.max()))
                )   
  
# initializing a line variable
line_pr, = axis.plot([], [], lw = 2, alpha=0.7, color='blue', label='preissmann')
line_df, = axis.plot([], [], lw = 2, alpha=0.7, color='orange', label='diff wave')

def init(): 
    line_pr.set_data([], [])
    line_df.set_data([], [])
    return line_pr, line_df

def animate_preiss_vs_dw(i):

    line_pr.set_data(x_real_distance, cwl_after_pr[i])
    line_df.set_data(x_real_distance, cwl_after_dw[i])
      
    return line_pr, line_df
   
anim_pr = FuncAnimation(fig, animate_preiss_vs_dw, init_func = init,
                     frames = NDAYS, interval = 100, blit = True)
anim_pr.save(F'preissmann_vs_diff-wave_n={N_MANNING}_q={EXTERNAL_SOURCE_q}_no_slope.mp4', writer = 'ffmpeg', fps = 30)


# %%
      
        
       