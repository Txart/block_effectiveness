#%%
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import matplotlib.cm as cm
import seaborn as sns
import rasterio
import copy
from tqdm import tqdm
import argparse

import read_weather_data

#%% Parseparser = argparse.ArgumentParser(description='Run 2d calibration')
parser = argparse.ArgumentParser(description='plot animations for a chosen set of parameters')

parser.add_argument('-p', '--param',
                    help='Number of set of params to animate',
                    type=int,
                    required=True)

args = parser.parse_args()

param_to_plot = args.param

#%% Funcs
def read_raster(fn):
    with rasterio.open(fn) as src:
        raster = src.read(1)
    return raster

def mask_raster_with_catchment_mask(raster):
    return np.ma.masked_where(raster < -1e10, raster)

def sample_raster_from_list_of_points(raster_filename:str, point_coords:np.ndarray) ->np.ndarray:
    with rasterio.open(raster_filename) as src:
        return np.array([value[0] for value in src.sample(point_coords)])
    
def map_coordinates_to_element_position_in_raster_array(list_of_coords, filename_raster):
    """Given a list of (lat,long) coordinates and a raster,
    return their corresponding positions in the raster array.

    Args:
        list_of_coords ([list or iterable]): Each element contains the (lat, long) pairs
        filename_raster ([string]): path to the reference raster file

    """ 	
    pos_raster_array = []
    with rasterio.open(filename_raster) as raster:
        for x,y in list_of_coords:
            px, py = raster.index(x, y)
            pos_raster_array.append((px, py))
            
    return pos_raster_array

def triage_wtd_files(weather_type, param_to_plot, blocked_status):

    noblocks_dry_directory = WTD_rasters_main_directory.joinpath("no_blocks_dry")
    noblock_dry_filenames = [noblocks_dry_directory.joinpath(f"zeta_after_{i}_DAYS.tif") for i in range(1, 366)]

    noblocks_wet_directory = WTD_rasters_main_directory.joinpath("no_blocks_wet")
    noblock_wet_filenames = [noblocks_wet_directory.joinpath(f"zeta_after_{i}_DAYS.tif") for i in range(1, 366)]

    yesblocks_dry_directory = WTD_rasters_main_directory.joinpath("yes_blocks_dry")
    yesblock_dry_filenames = [yesblocks_dry_directory.joinpath(f"zeta_after_{i}_DAYS.tif") for i in range(1, 366)]
    yesblocks_wet_directory = WTD_rasters_main_directory.joinpath("yes_blocks_wet")
    yesblock_wet_filenames = [yesblocks_wet_directory.joinpath(f"zeta_after_{i}_DAYS.tif") for i in range(1, 366)]
    if weather_type=='wet':
        if blocked_status == 'yesblocks':
            return noblock_wet_filenames
        elif blocked_status == 'noblocks':
            return yesblock_wet_filenames
         
    elif weather_type=='dry':
        if blocked_status == 'yesblocks':
            return noblock_dry_filenames
        elif blocked_status == 'noblocks':
            return  yesblock_dry_filenames

    raise ValueError('weather_type or blocked_status not recognized')

def triage_weather_files(weather_type):
    if weather_type=='wet':
        sourcesink = wet_sourcesink
    else:
        sourcesink = dry_sourcesink
    return sourcesink

#%% Directories
parent_directory = Path(".")
data_parent_folder = parent_directory.joinpath('data') 
output_folder = parent_directory.joinpath('output/plots')
WTD_rasters_main_directory = parent_directory.joinpath(f'output/params_number_{param_to_plot}') 
    
#%% Get P - ET
dry_sourcesink = read_weather_data.get_daily_net_source(year_type='elnino')
wet_sourcesink = read_weather_data.get_daily_net_source(year_type='normal')


#%% Choose what to plot
# Animations
PLOT_YES_AND_NO = True
PLOT_DIFF = False

# Static plots
PLOT_REPORT_PLOTS = False

# Global params
NDAYS = 365 # days to plot

#%% Animation with and without blocks over time
def anim_wtd(weather_type, param_to_plot):
    noblock_raster_filenames = triage_wtd_files(weather_type, param_to_plot, blocked_status='noblocks') 
    yesblock_raster_filenames = triage_wtd_files(weather_type, param_to_plot,  blocked_status='yesblocks') 

    sourcesink = triage_weather_files(weather_type)

    output_fn = output_folder.joinpath(f'{weather_type}_with_and_without_time_evolution_param_{param_to_plot}.mp4')

    if weather_type=='wet':
        out_fn = f'wet_diff_time_evolution_param_{param_to_plot}.mp4'
    else:
        out_fn = f'dry_diff_time_evolution_param_{param_to_plot}.mp4'
    # WTD max and min values for visualization
    MAX_ZETA = 1
    MIN_ZETA = -2
    # Set figure
    PLOTCOLS = 3
    PLOTROWS = 2
    FIGSIZE = (10,10)
    DPI = 300

    # Create fig layout
    fig, axes = plt.subplots(PLOTROWS, PLOTCOLS, dpi=DPI, figsize=FIGSIZE,
                            gridspec_kw={'width_ratios': [10, 10, 1],
                                        'height_ratios': [4,1]})
    gs = axes[1,1].get_gridspec()
    for ax in axes[1,:]: # remove axes in the bottom row
        ax.remove()
    ax_wtd = fig.add_subplot(gs[1, :]) # create axis in bottom row spanning all the columns
    ax_wtd.spines['right'].set_visible(False)
    ax_wtd.spines['top'].set_visible(False)
    ax_wtd.set_xlabel('time (days)')
    ax_wtd.set_ylabel('WTD (m)')

    ax_no = axes[0,0] # extend plot to axes 1 to 3
    ax_no.axis('off')
    ax_no.set_title('without blocks')
    
    ax_yes = axes[0,1]
    ax_yes.axis('off')
    ax_yes.set_title('with blocks')
    
    ax_bar =  axes[0,2]
    ax_bar.set_xlabel('WTD (m)')


    # Set colormap for raster figs
    CMAP = copy.copy(cm.Blues_r)
    CMAP.set_bad(color='white') # mask color


    # Plot static WTD at different points figure
    POINT_COORDS = [(399751.76044, 9769571.79923), (396751.76044, 9775571.79923), (408251.76044, 9780071.79923)]
    POINT_COLORS = ['tab:orange', 'tab:green', 'tab:red']
    wtd_yesblocks_at_points = np.zeros(shape=(NDAYS, len(POINT_COORDS)))
    wtd_noblocks_at_points = np.zeros(shape=(NDAYS, len(POINT_COORDS)))
    avg_wtd_yes = np.zeros(NDAYS)
    avg_wtd_no = np.zeros(NDAYS)
    
    for day in range(NDAYS):
        for point in POINT_COORDS:
            yes_filename = yesblock_raster_filenames[day]
            wtd_yesblocks_at_points[day] = sample_raster_from_list_of_points(raster_filename=yes_filename, point_coords=POINT_COORDS)
            no_filename = noblock_raster_filenames[day]
            wtd_noblocks_at_points[day] = sample_raster_from_list_of_points(raster_filename=no_filename, point_coords=POINT_COORDS)

            # average WTD
            yes_raster = read_raster(yes_filename)
            yes_masked_raster = mask_raster_with_catchment_mask(yes_raster)
            no_raster = read_raster(no_filename)
            no_masked_raster = mask_raster_with_catchment_mask(no_raster)
            
            avg_wtd_yes[day] = yes_masked_raster.mean()
            avg_wtd_no[day] = no_masked_raster.mean()

    # wtd at points
    for i, point in enumerate(POINT_COORDS):
        ax_wtd.plot(wtd_yesblocks_at_points[:,i], color=POINT_COLORS[i])
        ax_wtd.plot(wtd_noblocks_at_points[:,i], linestyle='dashed', color=POINT_COLORS[i])
        ax_wtd.fill_between(range(NDAYS),
                            y1=wtd_yesblocks_at_points[:,i],
                            y2=wtd_noblocks_at_points[:,i],
                            color=POINT_COLORS[i], alpha=0.2)
        
    # avg WTD
    ax_wtd.plot(avg_wtd_yes, label='avg with blocks', color='black')
    ax_wtd.plot(avg_wtd_no, label='avg without blocks', linestyle='dashed', color='black')
    ax_wtd.fill_between(range(NDAYS), y1=avg_wtd_yes, y2=avg_wtd_no, color='black', alpha=0.2)
    ax_wtd.legend()
    
    # Get positions of coordinates in array
    filename = noblock_raster_filenames[0] # any fn would do
    points_in_array = []
    points_in_array = map_coordinates_to_element_position_in_raster_array(list_of_coords=POINT_COORDS,
                                                                         filename_raster=filename)

    # MAIN ANIMATION LOOP. make list of images
    # Noblocks
    frames = [] # this is a list of images

    
    for day in tqdm(range(NDAYS)):
        # no blocks raster
        no_filename = noblock_raster_filenames[day]
        no_raster = read_raster(no_filename)
        no_masked_raster = mask_raster_with_catchment_mask(no_raster)
        no_image = ax_no.imshow(no_masked_raster, cmap=CMAP, animated=True,
                                        vmin=MIN_ZETA, vmax=MAX_ZETA)
        
        # yes blocks raster
        yes_filename = yesblock_raster_filenames[day]
        yes_raster = read_raster(yes_filename)
        yes_masked_raster = mask_raster_with_catchment_mask(yes_raster)
        yes_image = ax_yes.imshow(yes_masked_raster, cmap=CMAP, animated=True,
                                        vmin=MIN_ZETA, vmax=MAX_ZETA)

        # points in map
        for i, point in enumerate(points_in_array):
            ax_yes.scatter(x=point[1], y=point[0], marker='x', alpha=1, color=POINT_COLORS[i])
            ax_no.scatter(point[1], point[0], marker='x', alpha=1, color=POINT_COLORS[i])

        # colorbar
        cb = plt.colorbar(mappable=yes_image, cax=ax_bar)
        cb.outline.set_visible(False)
        
        # wtd line
        line_in_wtd = ax_wtd.axvline(x=day, ymin=0, ymax=1, color='black')
        
        # Append to list of frames
        frames.append([no_image, yes_image, line_in_wtd])
    
    print('Rasters read. Rendering (this may take a while)...')    
    
    ani = anim.ArtistAnimation(fig, frames, interval=50, blit=True,
                                    repeat_delay=1000)    
    
    ani.save(str(output_fn))

    print(f'Rendering done. Saved to: {output_fn}')

    return None

#%% Animation difference with and without over time
def anim_diff(weather_type, param_to_plot) -> None:
    noblock_raster_filenames = triage_wtd_files(weather_type, param_to_plot, blocked_status='noblock') 
    yesblock_raster_filenames = triage_wtd_files(weather_type, param_to_plot, blocked_status='yesblock') 

    sourcesink = triage_weather_files(weather_type)

    if weather_type=='wet':
        out_fn = f'wet_diff_time_evolution_param_{param_to_plot}.mp4'
    else:
        out_fn = f'dry_diff_time_evolution_param_{param_to_plot}.mp4'

    MAX_ZETA = 1
    MIN_ZETA = -1
    # Set figure
    PLOTCOLS = 2
    PLOTROWS = 2
    FIGSIZE = (10,10)
    DPI = 200

    # Create fig layout
    fig, axes = plt.subplots(PLOTROWS, PLOTCOLS, dpi=DPI, figsize=FIGSIZE,
                            gridspec_kw={'width_ratios': [20, 1],
                                        'height_ratios': [4,1]})
    fig.suptitle(f'blocks - no blocks, {weather_type} year. Parameter {param_to_plot}')
    gs = axes[1,1].get_gridspec()
    for ax in axes[1,:]: # remove axes in the bottom row
        ax.remove()
    ax_rain = fig.add_subplot(gs[1, :]) # create axis in bottom row spanning all the columns
    ax_rain.spines['right'].set_visible(False)
    ax_rain.spines['top'].set_visible(False)
    ax_rain.set_xlabel('time (d)')
    ax_rain.set_ylabel('P - ET (mm/day)')

    ax_diff = axes[0,0] # extend plot to axes 1 to 3
    ax_diff.axis('off')
    # ax_diff.set_title('With blocks - without blocks')
    
    ax_bar =  axes[0,1]
    ax_bar.set_xlabel('$\zeta$ difference (m)')


    # Set colormap for raster figs
    CMAP = copy.copy(cm.RdBu)
    CMAP.set_bad(color='white') # mask color

    # Plot static rainfall figure
    source_average_accross_weather_stations = sourcesink[:NDAYS]
    rain_image = ax_rain.bar(x=np.arange(NDAYS), height=source_average_accross_weather_stations)

    # make list of images
    # Noblocks
    frames = [] # this is a list of images

    print('Reading rasters...')
    for day in tqdm(range(NDAYS)):
        # difference raster
        no_filename = noblock_raster_filenames[day]
        no_raster = read_raster(no_filename)
        yes_filename = yesblock_raster_filenames[day]
        yes_raster = read_raster(yes_filename)
        diff_raster = yes_raster - no_raster
        masked_diff_raster = np.ma.masked_where(yes_raster < -1e10, diff_raster)

        diff_image = ax_diff.imshow(masked_diff_raster, cmap=CMAP, animated=True,
                                        vmin=MIN_ZETA, vmax=MAX_ZETA)
        
        # colorbar
        if day == 0:
            cb = plt.colorbar(mappable=diff_image, cax=ax_bar)
            cb.outline.set_visible(False)
        
        # rainfall
        line_in_rain = ax_rain.axvline(x=day, ymin=0, ymax=1, color='black')
        
        # Append to list of frames
        frames.append([diff_image, line_in_rain])
    
    print('Rasters read. Rendering (this may take a while)...')
        
    ani = anim.ArtistAnimation(fig, frames, interval=50, blit=True,
                                    repeat_delay=1000)

    output_fn = output_folder.joinpath(out_fn)
    ani.save(str(output_fn))

    print(f'Rendered and saved to {str(output_fn)}')

    return None

if PLOT_DIFF:
    anim_diff(weather_type='wet', param_to_plot=param_to_plot)
    anim_diff(weather_type='dry', param_to_plot=param_to_plot)

if PLOT_YES_AND_NO:
    anim_wtd(weather_type='wet', param_to_plot=param_to_plot)
    anim_wtd(weather_type='dry', param_to_plot=param_to_plot)

# %% Static plot of one point through a year. WITH BLOCKS.
if PLOT_REPORT_PLOTS:
    raise NotImplementedError('Not implemented for p2')
    sns.set_context("paper", font_scale=1.5)
    
    POINT_COORDS = [(399751.76044, 9769571.79923), (396751.76044, 9775571.79923), (408251.76044, 9780071.79923)]
    POINT_COLORS = ['tab:orange', 'tab:olive', 'tab:red']
    
    #######################
    # Map and wtd in 3 points
    
    fn_dem = r"C:\Users\03125327\Dropbox\PhD\Computation\ForestCarbon\2021 SMPP WTD customer work\qgis_derivated_data\dtm_depth_padded.tif"

    # Create fig
    fig, axes = plt.subplots(nrows=2, ncols=1, dpi=300, figsize=(20,10),
                            gridspec_kw={'height_ratios': [0.5*(1+np.sqrt(5)),1]})
    ax_map = axes[0]
    ax_wtd = axes[1]
    
    # Beautify axes
    ax_wtd.spines['right'].set_visible(False)
    ax_wtd.spines['top'].set_visible(False)
    ax_wtd.set_xlabel('Time (days)')
    ax_wtd.set_ylabel('WTL (m)')

    ax_map.axis('off')
    
    # Set colormap for raster figs
    CMAP = copy.copy(cm.Blues_r)
    CMAP.set_bad(color='white') # mask color
    
    # Plot DEM raster
    dem = read_raster(fn_dem)
    dem_masked_raster = mask_raster_with_catchment_mask(dem)
    ax_map.imshow(dem_masked_raster, cmap=CMAP)
    
    # points in dem
    points_in_array = map_coordinates_to_element_position_in_raster_array(list_of_coords=POINT_COORDS,
                                                                         filename_raster=fn_dem)
    for i, point in enumerate(points_in_array):
        ax_map.scatter(x=point[1], y=point[0], marker='x', s=100, alpha=1, color=POINT_COLORS[i])

    # Get WTD at chosen points
    wtd_yesblocks_at_points = np.zeros(shape=(NDAYS, len(POINT_COORDS)))
    for day, yes_filename in enumerate(yesblock_raster_filenames):
        for point in POINT_COORDS:
            wtd_yesblocks_at_points[day] = sample_raster_from_list_of_points(raster_filename=yes_filename, point_coords=POINT_COORDS)

    # plot wtd at points
    for i, point in enumerate(POINT_COORDS):
        ax_wtd.plot(wtd_yesblocks_at_points[:,i], color=POINT_COLORS[i])

#%%
    ##########################
    # Plot measured vs modelled
    # Get sensor data
    df_wtd_other = pd.read_csv(Path.joinpath(data_parent_folder, 'WTL_Other_2021.csv'), sep=';')
    df_wtd_transects = pd.read_csv(Path.joinpath(data_parent_folder, 'WTL_PatPos_2021.csv'), sep=';')
    df_wtd_transects_canal = pd.read_csv(Path.joinpath(data_parent_folder, 'WaterLevel_Adjusted_2021.csv'), sep=';')
    
    # Sensor coords
    df_dipwell_loc = pd.read_csv(Path.joinpath(data_parent_folder, 'all_dipwell_coords_projected.csv'))
    dipwell_coords = {}
    for _, row in df_dipwell_loc.iterrows():
        dipwell_coords[row.ID] = (row.x, row.y)
        
    small_names = ['MC', 'SI', 'SE', 'LE', 'RA']
    
    for tname in small_names:
        dipwell_coords[tname + '.0'] = dipwell_coords[tname + '.1']

    dict_of_transects = {}
    for small_name in small_names:
        df_wtd_transects[small_name + '.0'] = df_wtd_transects_canal[small_name]
        all_col_names_with_name = [name for name in df_wtd_transects.columns if small_name in name]
        dict_of_transects[small_name] = df_wtd_transects[all_col_names_with_name]
        all_dfs = list(dict_of_transects.values()) + [df_wtd_other]
        all_sensor_WTD = pd.concat(all_dfs, axis=1)
        
    # remove Date column
    all_sensor_WTD.drop(labels='Date', axis='columns', inplace=True)

    # Units: cm -> m
    all_sensor_WTD = all_sensor_WTD/100
    
    # Drop WTD sensor columns from dataframe whose position is unknown 
    sensor_names_with_unknown_coords = [sname for sname in list(all_sensor_WTD.columns) if sname not in dipwell_coords]
    all_sensor_WTD.drop(sensor_names_with_unknown_coords, axis='columns', inplace=True)
    
    sensor_names = list(all_sensor_WTD.columns)
    sensor_coords = [dipwell_coords[sn] for sn in sensor_names]

    # Remove outliers from dipwell measurements
    from scipy.stats import zscore
    for colname, values in all_sensor_WTD.iteritems():
        no_nan_indices = values.dropna().index
        zscore_more_than_3std = (np.abs(zscore(values.dropna())) > 3)
        # If value above 3 STD, set it to NaN
        for i, index in enumerate(no_nan_indices):
            if zscore_more_than_3std[i]:
                all_sensor_WTD.loc[index,colname] = np.nan

    # Get modelled values at sensor positions
    def create_modelled_result_dataframe(days_list, folder_path):
        # Save zeta values at sensors in a dataframe of the same format as the measurements df
        modelled_WTD_at_sensors = pd.DataFrame().reindex_like(all_sensor_WTD)
        for day in days_list:
            try:
                zeta_fn = Path.joinpath(folder_path, f"zeta_after_{int(day)}_DAYS.tif")
                zeta_values_at_sensor_locations = sample_raster_from_list_of_points(raster_filename=zeta_fn, point_coords=sensor_coords)
                modelled_WTD_at_sensors.loc[day-1] = zeta_values_at_sensor_locations
            except Exception as e:
                print(e)
                break
                
        return modelled_WTD_at_sensors
    
    modeled_yesblocks = create_modelled_result_dataframe(days_list=range(1, NDAYS+1), folder_path=yesblocks_rasters_directory)

    # Plot
    fig, ax = plt.subplots(figsize=(16,9),dpi=500)
    ax.set_ylabel('WTL (m)')
    ax.set_xlabel('Time (days)')
    
    # Plot measured
    all_sensor_WTD.plot(legend=False, marker='.', linestyle='None',  ax=ax, alpha= 0.3, color='grey')
    
    # Plot modelled
    modeled_yesblocks.plot(ax=ax, legend=False)

# %%

