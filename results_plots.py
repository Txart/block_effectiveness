# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib import cm
import matplotlib
import rasterio
from pathlib import Path
from collections import OrderedDict
import seaborn as sns
import copy

import read_weather_data

# %% Define style
# plt.style.use('seaborn-paper')
# plt.rc('font', family='serif')
# sns.set_context('paper', font_scale=1.5)

plt.style.use('paper_plot_style.mplstyle')

# %% Params

N_DAYS = 365

params_to_plot = [11, 12, 13, 14]
N_PARAMS = len(params_to_plot)

parent_folder = Path('.')
output_folder = Path('output/plots')
data_folder = Path('output')
# data_folder = Path('hydro_for_plots')

# colormap
# see https://matplotlib.org/3.5.0/tutorials/colors/colormaps.html
cmap = cm.get_cmap('Dark2')
cmap_parameterization = plt.get_cmap('Set1')
cmap_wtd_raster = copy.copy(plt.get_cmap('Blues_r')) # reversed blues  
cmap_diff = copy.copy(cm.RdBu)

param_colors = [cmap_parameterization(i) for i in range(len(params_to_plot))]

# Foldernames
modes = ([
    {'foldername': 'no_blocks_dry',
        'name': 'el Niño, no blocks',
        'weather': 'dry',
        'blocks': 'no blocks',
        'color': cmap(1),
        'marker': 'x',
        'linestyle': 'dashed'},
    {'foldername': 'no_blocks_wet',
        'name': 'wet, no blocks',
        'weather': 'wet',
        'blocks': 'no blocks',
        'color': cmap(0),
        'marker': 'x',
        'linestyle': 'dashed'},
    {'foldername': 'yes_blocks_dry',
        'name': 'el Niño, blocks',
        'weather': 'dry',
        'blocks': 'blocks',
        'color': cmap(1),
        'marker': 'o',
        'linestyle': 'solid'},
    {'foldername': 'yes_blocks_wet',
        'name': 'wet, blocks',
        'weather': 'wet',
        'blocks': 'blocks',
        'color': cmap(0),
        'marker': 'o',
        'linestyle': 'solid'},
    # {'foldername': 'yes_blocks',
    # 'name':'weather_station, blocks',
    # 'color':'black',
    # 'marker':'.',
    # 'linestyle':'solid'}
])
# %% Useful funcs


def plot_repeated_labels_only_once_in_legend():
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    return None


# %% Read data
# Get raster dimensions from first raster file
template_rfn = data_folder.joinpath(
    f'params_number_{params_to_plot[0]}/{modes[0]["foldername"]}/zeta_after_1_DAYS.tif')
with rasterio.open(template_rfn, 'r') as src:
    template_raster = src.read(1)
raster_shape = template_raster.shape


def read_data(n_params, modes, n_days, rasters_shape):
    # data = [[[0]*N_DAYS for _ in range(len(mode_foldernames))] for _ in range(N_PARAMS)]
    data = np.zeros(shape=(len(n_params), len(modes), n_days,
                    rasters_shape[0], rasters_shape[1]))
    for i_param, n_param in enumerate(n_params):
        param_folder = data_folder.joinpath(f'params_number_{n_param}')
        for n_mode, mode_foldername in enumerate([mode['foldername'] for mode in modes]):
            foldername = param_folder.joinpath(mode_foldername)
            for day in range(0, N_DAYS):
                fname = foldername.joinpath(
                    f'zeta_after_{int(day+1)}_DAYS.tif')
                with rasterio.open(fname, 'r') as src:
                    data[i_param][n_mode][day] = src.read(1)
    return data


# Array containing data mimicks folder structure
# Structure: [param_number, block_and_precip_type, day_number, raster]
data = read_data(params_to_plot, modes, N_DAYS, raster_shape)

# Change raster nans to zeros
data_with_nan = data[:]
data = np.nan_to_num(data, nan=0.0)
#%% P minus ET
dry_sourcesink = read_weather_data.get_daily_net_source(year_type='elnino')
wet_sourcesink = read_weather_data.get_daily_net_source(year_type='normal')
# mm to m
dry_sourcesink, wet_sourcesink = 1000*dry_sourcesink, 1000*wet_sourcesink
# %% Compute averages
# Axes:
# 0:param_number
# 1:block_and_precip_type
# 2:day_number
# (3,4): raster
spatial_average = np.mean(data, axis=(3, 4))
temporal_average = np.mean(data, axis=2)
spatiotemporal_average = np.mean(spatial_average, axis=2)

# cap wtd at zero for CO2 emission calculation
data_capped_at_zero = data[:]
data_capped_at_zero[data > 0] = 0.0

# CO2 emission calc. Params from Jauhiainen 2012
def mCO2_per_hectare_per_day_from_zeta(zeta):
    # zeta must be daily data
    alpha = 74.11/365  # Mg/ha/m/day
    beta = 29.34/365  # Mg/ha/day
    # m_CO2_daily = -alpha * zeta + beta 
    return -alpha * zeta + beta 

m_CO2_daily = mCO2_per_hectare_per_day_from_zeta(zeta = np.mean(data_capped_at_zero, axis=(3, 4)))  # Mg/ha/day
cum_CO2_daily = np.cumsum(m_CO2_daily, axis=2)

# unblocked - blocked
sequestered_wet_CO2_daily = m_CO2_daily[:, 1, :] - m_CO2_daily[:, 3, :]
sequestered_dry_CO2_daily = m_CO2_daily[:, 0, :] - m_CO2_daily[:, 2, :]
cum_sequestered_wet_CO2_daily = np.cumsum(sequestered_wet_CO2_daily, axis=1)
cum_sequestered_dry_CO2_daily = np.cumsum(sequestered_dry_CO2_daily, axis=1)

# Translate numpy array to pandas dataframe for various plots
# spatial average
df_spatialaverage_cols = ['avg_zeta', 'weather', 'param', 'blocks', 'mode']
df_spatialaverage = pd.DataFrame(columns=df_spatialaverage_cols)
for i_param, param_name in enumerate(params_to_plot):
    for i_mode, mode in enumerate(modes):
        arr = np.zeros(
            shape=(N_DAYS, len(df_spatialaverage_cols)), dtype=object)
        for i, avg_zeta in np.ndenumerate(spatial_average[i_param, i_mode, :]):
            arr[i, 0] = avg_zeta
            arr[i, 2] = param_name
            arr[i, 3] = 'no blocks' if i_mode == 0 or i_mode == 1 else 'blocks'
            arr[i, 4] = mode['name']
            if i_mode == 1 or i_mode == 3:
                arr[i, 1] = 'wet'
            elif i_mode == 0 or i_mode == 2:
                arr[i, 1] = 'dry'
            else:
                arr[i, 1] = 'weather_station'
        new_df = pd.DataFrame(arr, columns=df_spatialaverage_cols)
        df_spatialaverage = pd.concat(
            [df_spatialaverage, new_df], ignore_index=True)
df_spatialaverage['avg_zeta'] = df_spatialaverage['avg_zeta'].astype('float')

# Daily spatial averages
df_daily_spatialaverage_cols = [
    'day', 'avg_zeta', 'weather', 'param', 'blocks', 'mode']
df_daily_spatialaverage = pd.DataFrame(columns=df_daily_spatialaverage_cols)
for i_param, param_name in enumerate(params_to_plot):
    for i_mode, mode in enumerate(modes):
        arr = np.zeros(
            shape=(N_DAYS, len(df_daily_spatialaverage_cols)), dtype=object)
        for i, avg_zeta in np.ndenumerate(spatial_average[i_param, i_mode, :]):
            arr[i, 0] = i
            arr[i, 1] = avg_zeta
            arr[i, 3] = param_name
            arr[i, 4] = 'no blocks' if i_mode == 0 or i_mode == 1 else 'blocks'
            arr[i, 5] = mode['name']
            if i_mode == 1 or i_mode == 3:
                arr[i, 2] = 'wet'
            elif i_mode == 0 or i_mode == 2:
                arr[i, 2] = 'dry'
            else:
                arr[i, 2] = 'weather_station'
        new_df = pd.DataFrame(arr, columns=df_daily_spatialaverage_cols)
        df_daily_spatialaverage = pd.concat(
            [df_daily_spatialaverage, new_df], ignore_index=True)
df_daily_spatialaverage['avg_zeta'] = df_daily_spatialaverage['avg_zeta'].astype(
    'float')
df_daily_spatialaverage['day'] = df_daily_spatialaverage['day'].astype('int')

# CO2 dataframe
co2_columns = ['day', 'sequestered_CO2',
               'cum_sequestered_CO2', 'weather', 'param']
df_daily_dry_CO2 = pd.DataFrame(
    columns=co2_columns, index=range(N_DAYS * N_PARAMS))
df_daily_wet_CO2 = pd.DataFrame(
    columns=co2_columns, index=range(N_DAYS * N_PARAMS))

new_df = pd.DataFrame(columns=co2_columns, index=range(N_DAYS))

for i_param, param_name in enumerate(params_to_plot):
    # dry
    new_df['day'] = np.arange(1, N_DAYS+1)
    new_df['sequestered_CO2'] = sequestered_dry_CO2_daily[i_param]
    new_df['cum_sequestered_CO2'] = cum_sequestered_dry_CO2_daily[i_param]
    new_df['weather'] = 'dry'
    new_df['param'] = param_name
    df_daily_dry_CO2 = pd.concat([df_daily_dry_CO2, new_df], ignore_index=True)
    # wet
    new_df['day'] = np.arange(1, N_DAYS+1)
    new_df['sequestered_CO2'] = sequestered_wet_CO2_daily[i_param]
    new_df['cum_sequestered_CO2'] = cum_sequestered_wet_CO2_daily[i_param]
    new_df['weather'] = 'wet'
    new_df['param'] = param_name
    df_daily_wet_CO2 = pd.concat([df_daily_wet_CO2, new_df], ignore_index=True)

# %% Compute differences
dry_diffs = data[:,2,:,:,:] - data[:,0,:,:,:] 
wet_diffs = data[:,3,:,:,:] - data[:,1,:,:,:] 
dry_diffs_with_nan = data_with_nan[:,2,:,:,:] - data_with_nan[:,0,:,:,:] 
wet_diffs_with_nan = data_with_nan[:,3,:,:,:] - data_with_nan[:,1,:,:,:] 

dry_diffs_spatial_avg = np.mean(dry_diffs, axis=(-2, -1))
wet_diffs_spatial_avg = np.mean(wet_diffs, axis=(-2, -1))


all_avg_dry_diffs_with_nan = np.mean(dry_diffs_with_nan, axis=(0,1))
all_avg_wet_diffs_with_nan = np.mean(wet_diffs_with_nan, axis=(0,1))

all_avg_diffs = 0.5 * (all_avg_dry_diffs_with_nan + all_avg_wet_diffs_with_nan) 

# %% Compute Spatial extent of block effects
import geopandas as gpd
import rasterio
from scipy.spatial import distance_matrix
# block positions
fn_block_positions = Path(r"C:\Users\03125327\github\paper2\data\new_area\dams.gpkg")
fn_dtm = Path(r"C:\Users\03125327\github\paper2\data\new_area\new_area_dtm.tif")

block_gdf = gpd.read_file(fn_block_positions)
block_gdf.to_crs(crs='EPSG:32748', inplace=True)
block_x_coords = block_gdf.geometry.x.to_numpy()
block_y_coords = block_gdf.geometry.y.to_numpy()

with rasterio.open(fn_dtm) as src:
    dtm_array = src.read(1)
    pixel_size = src.transform[0]
    top_left_corner_coords = src.transform * (0,0)
    bottom_right_corner_coords = src.transform * (src.width, src.height)

x_indices = np.arange(start=top_left_corner_coords[0],
                      stop=bottom_right_corner_coords[0],
                      step=pixel_size)
y_indices = np.arange(start=bottom_right_corner_coords[1],
                      stop=top_left_corner_coords[1],
                      step=pixel_size)
catchment_x, catchment_y = np.meshgrid(x_indices, y_indices)

block_coordinates = np.array([block_x_coords, block_y_coords]).T  
catchment_coordinates = np.array([catchment_x, catchment_y]).T
newshape = (catchment_coordinates.shape[0]*catchment_coordinates.shape[1], 2)
catchment_coordinates_flat = catchment_coordinates.reshape(newshape)
dist_mat = distance_matrix(block_coordinates, catchment_coordinates_flat)
min_dist = np.min(dist_mat, axis=0)
min_dist_array_to_blocks = min_dist.reshape(catchment_coordinates.shape[0], catchment_coordinates.shape[1]).T[::-1]
# min_dist_array_to_blocks contains the shortest distance to a block for each pixel. Try:
# plt.imshow(min_dist_array); plt.colorbar()
# %% Compute binned mean distance
BIN_WIDTH_IN_METERS = 20
bins = np.arange(start=0, stop=np.max(min_dist_array_to_blocks), step=BIN_WIDTH_IN_METERS)
bin_indices = np.digitize(min_dist_array_to_blocks.ravel(), bins=bins)

dry_diffs_temporal_avg = np.mean(dry_diffs, axis=(1))
wet_diffs_temporal_avg = np.mean(wet_diffs, axis=(1))

wet_mean_diff_binned = np.zeros(shape=(N_PARAMS, len(bins)))
dry_mean_diff_binned = np.zeros(shape=(N_PARAMS, len(bins)))
for i_param, _ in enumerate(params_to_plot):
    for i_bin, _ in enumerate(bins):
        wet_mean_diff_binned[i_param, i_bin] = np.mean(wet_diffs_temporal_avg[i_param,:,:].ravel()[bin_indices==i_bin+1])
        dry_mean_diff_binned[i_param, i_bin] = np.mean(dry_diffs_temporal_avg[i_param,:,:].ravel()[bin_indices==i_bin+1])


#%% --------------------------------
# Plot All averaged into one WTD raster
# -----------------------------------

MIN_DIFF = -1.0
MAX_DIFF = 1.0
cmap_diff.set_bad(color='white') # mask color
plt.figure(figsize=(4,4), dpi=600)
plt.axis('off')
plt.title('Diffs avged over everything', fontsize=10)
im = plt.imshow(all_avg_diffs, cmap=cmap_diff,
           vmin=MIN_DIFF, vmax=MAX_DIFF)
cb = plt.colorbar(im, shrink=0.7)
cb.ax.tick_params(labelsize=10)
plt.savefig(output_folder.joinpath(f'diff_averaged_over_everything.png'),
                bbox_inches='tight')
plt.show()
# %%------------------------------
# Every param spatial_average vs time, wet and dry
# --------------------------------
fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(9,12))
ax_wet, ax_dry = axes
ax_wet.set_title('Wet')
ax_dry.set_title('Dry')
ax_dry.set_xlabel('time (d)')
ax_dry.set_ylabel(r'$\langle\zeta\rangle (m)$')
ax_wet.set_ylabel(r'$\langle\zeta\rangle (m)$')
for ax in axes:
    ax.grid(visible='True')

for i_mode, mode in enumerate(modes):
    for i_param, param_to_plot in enumerate(params_to_plot):
        if i_mode == 0 or i_mode == 2: # dry
            # plot lines
            ax_dry.plot(range(N_DAYS), spatial_average[i_param, i_mode],
                    label=mode['name'],
                    color=param_colors[i_param],
                    linestyle=mode['linestyle']) 
            # fill diff with color 
            ax_dry.fill_between(x=range(N_DAYS),
                    y1=spatial_average[i_param, 2],
                    y2=spatial_average[i_param, 0],
                    color=param_colors[i_param],
                    alpha=0.1)

        if i_mode == 1 or i_mode == 3: # wet
            # plot lines
            ax_wet.plot(range(N_DAYS), spatial_average[i_param, i_mode],
                    label=mode['name'],
                    color=param_colors[i_param],
                    linestyle=mode['linestyle']) 
            # fill diff with color 
            ax_wet.fill_between(x=range(N_DAYS),
                    y1=spatial_average[i_param, 3],
                    y2=spatial_average[i_param, 1],
                    color=param_colors[i_param],
                    alpha=0.1)
# Plot custom legend
color_legend_elements = [Patch(facecolor=pc, edgecolor=pc, label=f'param {i+1}') for i,pc in enumerate(param_colors)] 
linestyle_legend_elements = [Line2D([],[], color='black', lw=2, linestyle='solid', label='blocked'),
                                Line2D([],[], color='black', lw=2, linestyle='dashed', label='not blocked')]
legend_elements = color_legend_elements + linestyle_legend_elements
ax_wet.legend(handles=legend_elements)

fig.tight_layout()
plt.savefig(output_folder.joinpath(f'every_param_wet_and_dry'),
            bbox_inches='tight')
plt.show()

# Every param cumulative diff over time, wet and dry
fig, axes = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=(4*1.69,4))
ax_wet, ax_dry = axes
ax_wet.set_title('Wet', fontsize=12)
ax_dry.set_title('Dry', fontsize=12)
ax_dry.set_xlabel('time (d)')
ax_wet.set_xlabel('time (d)')
ax_wet.set_ylabel(r'$\langle \Delta \zeta\rangle (m)$')

# Set labels for subplots
for ax, label in zip(axes.flat, ["a)", "b)"]):
    ax.text(0 + 0.02, 1 - 0.02, s=label,
                transform=ax.transAxes,
                verticalalignment='top',
                fontsize=10,
                bbox=dict(facecolor='1.0', edgecolor='none', pad=3.0))
for ax in axes:
    ax.grid(visible='True')

for i_param, param_to_plot in enumerate(params_to_plot):
    # plot dry lines
    ax_dry.plot(range(N_DAYS), dry_diffs_spatial_avg[i_param, :],
            label=f'param {param_to_plot}',
            color=param_colors[i_param])
    # plot wet lines
    ax_wet.plot(range(N_DAYS), wet_diffs_spatial_avg[i_param, :],
            label=f'param {param_to_plot}',
            color=param_colors[i_param])
    
# Plot net rainfall in a twin axis
ax_dry_twin = ax_dry.twinx()
ax_wet_twin = ax_wet.twinx()
ax_wet_twin.get_yaxis().set_ticks([])
# share the secondary axes
ax_wet_twin.get_shared_y_axes().join(ax_wet_twin, ax_dry_twin)
ax_wet_twin.bar(x=range(len(wet_sourcesink)),
        height=wet_sourcesink,
        label='wet year',
        alpha=0.4,
        color=modes[1]['color']
        )
ax_dry_twin.bar(x=range(len(dry_sourcesink)),
        height=dry_sourcesink,
        label='dry year',
        alpha=0.4,
        color=modes[0]['color']
        )
ax_dry_twin.set_ylim(200, -20)
ax_dry_twin.set_ylabel(r'$ P - ET (mm/d)$')



ax_wet.legend()

fig.tight_layout()
plt.savefig(output_folder.joinpath(f'every_param_diff_vs_time_wet_and_dry.png'),
            bbox_inches='tight')
plt.show()
# %%------------------------------
# COMBINED Every param spatial_average vs time, wet and dry
# --------------------------------
fig, axes = plt.subplots(nrows=2, ncols=2,
                         sharex="col", sharey="row",
                         figsize=(5.31, 5.31), constrained_layout=True)
ax_wet_wtd, ax_dry_wtd = axes[0]
ax_wet_diff, ax_dry_diff = axes[1]
ax_wet_wtd.set_title('Wet', fontsize=13)
ax_dry_wtd.set_title('Dry', fontsize=13)
ax_dry_diff.set_xlabel('time (d)')
ax_wet_diff.set_xlabel('time (d)')
ax_dry_diff.set_xticks([0, 100, 200, 300])
ax_wet_diff.set_xticks([0, 100, 200, 300])
ax_wet_wtd.set_ylabel(r'$\langle\zeta\rangle (m)$')
ax_wet_diff.set_ylabel(r'$\langle \Delta \zeta\rangle (m)$')
                       
for ax in axes.flat:
    ax.grid(visible='True', linewidth=0.5)

# Set labels for subplots
for ax, label in zip(axes.flat, ["a)", "b)", "c)", "d)"]):
    ax.text(0 + 0.04, 1 - 0.04, s=label,
                transform=ax.transAxes,
                verticalalignment='top',
                fontsize=10,
                bbox=dict(facecolor='1.0', edgecolor='none', pad=3.0))

for i_mode, mode in enumerate(modes):
    for i_param, param_to_plot in enumerate(params_to_plot):
        if i_mode == 0 or i_mode == 2: # dry
            # plot lines
            ax_dry_wtd.plot(range(N_DAYS), spatial_average[i_param, i_mode],
                    label=mode['name'],
                    color=param_colors[i_param],
                    linestyle=mode['linestyle'],
                    linewidth=0.5) 
            # fill diff with color 
            ax_dry_wtd.fill_between(x=range(N_DAYS),
                    y1=spatial_average[i_param, 2],
                    y2=spatial_average[i_param, 0], lw=1,
                    color=param_colors[i_param],
                    alpha=0.1)

        if i_mode == 1 or i_mode == 3: # wet
            # plot lines
            ax_wet_wtd.plot(range(N_DAYS), spatial_average[i_param, i_mode],
                    label=mode['name'],
                    color=param_colors[i_param],
                    linestyle=mode['linestyle'], 
                    linewidth=0.5) 
            # fill diff with color 
            ax_wet_wtd.fill_between(x=range(N_DAYS),
                    y1=spatial_average[i_param, 3],
                    y2=spatial_average[i_param, 1],
                    color=param_colors[i_param],
                    alpha=0.1)
# Plot custom legend
color_legend_elements = [Patch(facecolor=pc, edgecolor=None, label=f'param {i+1}', linewidth=0) for i,pc in enumerate(param_colors)] 
linestyle_legend_elements = [Line2D([],[], color='black', lw=1, linestyle='solid', label='blocked'),
                                Line2D([],[], color='black', lw=1, linestyle='dashed', label='not blocked')]
legend_elements = color_legend_elements + linestyle_legend_elements
ax_wet_wtd.legend(handles=legend_elements, fontsize=7)

for i_param, param_to_plot in enumerate(params_to_plot):
    # plot dry lines
    ax_dry_diff.plot(range(N_DAYS), dry_diffs_spatial_avg[i_param, :],
            label=f'param {param_to_plot}',
            linewidth=0.5,
            color=param_colors[i_param])
    # plot wet lines
    ax_wet_diff.plot(range(N_DAYS), wet_diffs_spatial_avg[i_param, :],
            label=f'param {param_to_plot}',
            linewidth=0.5,
            color=param_colors[i_param])
    
# Plot net rainfall in a twin axis
ax_dry_diff_twin = ax_dry_diff.twinx()
ax_wet_diff_twin = ax_wet_diff.twinx()
ax_wet_diff_twin.get_yaxis().set_ticks([])
# share the secondary axes
ax_wet_diff_twin.get_shared_y_axes().join(ax_wet_diff_twin, ax_dry_diff_twin)
ax_wet_diff_twin.bar(x=range(len(wet_sourcesink)),
        height=wet_sourcesink,
        label='wet year',
        alpha=0.4,
        color=modes[1]['color']
        )
ax_dry_diff_twin.bar(x=range(len(dry_sourcesink)),
        height=dry_sourcesink,
        label='dry year',
        alpha=0.4,
        color=modes[0]['color']
        )
ax_dry_diff_twin.axhline(y=0, color='black', alpha=0.3, ls='dashed', lw=0.5)
ax_wet_diff_twin.axhline(y=0, color='black', alpha=0.3, ls='dashed', lw=0.5)
ax_dry_diff_twin.set_ylim(200, -10)
ax_dry_diff_twin.set_yticks(ax_dry_diff_twin.get_yticks()[1:-2]) # Remove some ticks from rainfall
ax_dry_diff_twin.set_ylabel(r'$ P - ET (mm/d)$')

plt.savefig(output_folder.joinpath(f'COMBINED_every_param_diff_vs_time_wet_and_dry.png'),
            bbox_inches='tight')
plt.show()



# %%------------------------------
# One param spatial_average vs time
# --------------------------------
for i_param, param_to_plot in enumerate(params_to_plot):
    plt.figure()
    plt.title(f'parameter number {param_to_plot}')
    plt.xlabel('time (d)')
    plt.ylabel(r'$\langle\zeta\rangle (m)$')
    for i_mode, mode in enumerate(modes):
        plt.plot(range(N_DAYS), spatial_average[i_param, i_mode],
                 label=mode['name'],
                 color=mode['color'],
                 linestyle=mode['linestyle'])
    # fill diff with color
    # dry year
    plt.fill_between(x=range(N_DAYS),
                     y1=spatial_average[i_param, 2],
                     y2=spatial_average[i_param, 0],
                     color=modes[0]['color'],
                     alpha=0.1)
    # wet year
    plt.fill_between(x=range(N_DAYS),
                     y1=spatial_average[i_param, 3],
                     y2=spatial_average[i_param, 1],
                     color=modes[1]['color'],
                     alpha=0.1)
    plt.legend()
    plt.savefig(output_folder.joinpath(f'avg_zeta_vs_time_param_{param_to_plot}'),
                bbox_inches='tight')
    plt.show()

# -----------------------------------------
# One param cumulative saved WTD over time
# -----------------------------------------

for i_param, param_to_plot in enumerate(params_to_plot):
    plt.figure()
    plt.title(f'parameter number {param_to_plot}')
    plt.xlabel('time (d)')
    plt.ylabel('cum CO2 emissions $(Mg ha^{-1})$')
    plt.plot(range(N_DAYS), cum_sequestered_dry_CO2_daily[i_param],
             label='el Niño',
             color=modes[0]['color']
             )
    plt.plot(range(N_DAYS), cum_sequestered_wet_CO2_daily[i_param],
             label='wet',
             color=modes[1]['color']
             )
    plt.legend()
    plt.savefig(output_folder.joinpath(f'daily_cumulative_CO2_param_{param_to_plot}'),
                bbox_inches='tight')
    plt.show()

# ---------------------------------------
# All params spatial average vs days
# --------------------------------------
fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(9,12))
ax_wet, ax_dry = axes
ax_wet.set_title('All parameters, wet, daily avg zeta over time')
p = sns.lineplot(
    data=df_daily_spatialaverage[df_daily_spatialaverage['weather'] == 'wet'],
    x='day', y='avg_zeta',
    style='blocks', style_order=(['blocks', 'no blocks']),
    ci='sd',
    ax=ax_wet,
    color=modes[1]['color'],
)
p.set_ylabel(r'$\langle\zeta\rangle (m)$')
plot_repeated_labels_only_once_in_legend()
plt.savefig(output_folder.joinpath(f'avg_zeta_vs_time_WET_all_params'),
            bbox_inches='tight')

ax_dry.set_title('All parameters, dry, daily avg zeta over time')
p = sns.lineplot(
    data=df_daily_spatialaverage[df_daily_spatialaverage['weather'] == 'dry'],
    x='day', y='avg_zeta',
    style='blocks', style_order=(['blocks', 'no blocks']),
    color=modes[2]['color'],
    ax=ax_dry,
    ci='sd')
p.set_xlabel('time (d)')
p.set_ylabel(r'$\langle\zeta\rangle (m)$')
plot_repeated_labels_only_once_in_legend()
plt.savefig(output_folder.joinpath(f'avg_zeta_vs_time_wet_and_dry_all_params'),
            bbox_inches='tight')
plt.show()


# -----------------------------------------
# Avg params cumulative saved CO2 over time
# -----------------------------------------
fig, ax = plt.subplots()
ax.grid(visible=True, linewidth=0.5)
ax.set_title(f'All parameters, sequestered CO_2')
p = sns.lineplot(
    data=df_daily_dry_CO2,
    x='day', y='cum_sequestered_CO2',
    label='dry',
    linewidth=0.8,
    color=modes[2]['color'],
    ci='sd',
    ax=ax
)
p = sns.lineplot(
    data=df_daily_wet_CO2,
    x='day', y='cum_sequestered_CO2',
    label='wet',
    linewidth=0.8,
    color=modes[3]['color'],
    ci='sd',
    ax=ax
)
p.set_xlabel('time (d)')
p.set_ylabel('sequestered $CO_2 (Mg ha^{-1})$')
plt.legend(loc='upper left')
plt.savefig(output_folder.joinpath(f'daily_cumulative_CO2_all_params'),
            bbox_inches='tight')

plt.show()

# ------------------------------------------
# All params cumulativesaved CO2 over time
# -----------------------------------------
fig, ax = plt.subplots()
ax.grid(visible=True, axis='y', linewidth=0.5)
ax.set_frame_on(False)
ax.set_title('Sequestered $CO_2$ per parameter')
ax.set_ylabel('sequestered $CO_2 (Mg ha^{-1})$')
ax.set_xlabel('time (d)')

for ip, p in enumerate(params_to_plot):
    df = df_daily_dry_CO2[df_daily_dry_CO2['param'] == p]
    ax.plot(df.day, df.cum_sequestered_CO2,
            color=param_colors[ip],
            label=f'parameter {p}',
            linewidth=0.8,
            linestyle='dashed')
    df = df_daily_wet_CO2[df_daily_dry_CO2['param'] == p]
    ax.plot(df.day, df.cum_sequestered_CO2,
            color=param_colors[ip],
            label=f'parameter {p}',
            linewidth=0.8,
            linestyle='solid')

# Plot custom legend
color_legend_elements = [Line2D([], [], color=pc, lw=1, label=f'param {i+1}') for i,pc in enumerate(param_colors)] 
linestyle_legend_elements = [Line2D([],[], color='black', lw=1, linestyle='solid', label='wet'),
                                Line2D([],[], color='black', lw=1, linestyle='dashed', label='dry')]
legend_elements = color_legend_elements + linestyle_legend_elements
ax.legend(handles=legend_elements, fontsize=8)
plt.savefig(output_folder.joinpath(f'all_params_daily_cumulative_CO2.png'),
            bbox_inches='tight')
plt.show()

# ---------------------------------------
# %% Modes spatiotemporal avg per param
# ---------------------------------------
plt.figure()
for i_param, param_name in enumerate(params_to_plot):
    # with plt.style.context(("seaborn-paper",)):
    for i_mode, mode in enumerate(modes):
        plt.scatter(
            i_param + 1, spatiotemporal_average[i_param, i_mode],
            label=mode['name'],
            marker=mode['marker'],
            color=mode['color'])
plt.xlabel('parameter number')
# Plot lablels only once
plot_repeated_labels_only_once_in_legend()
plt.show()

# ---------------------------------------
# Violinplots
# ---------------------------------------

# Drop data with weather_station weather from plot
df_spatialaverage = df_spatialaverage[df_spatialaverage.weather !=
                                      'weather_station']

# Show means
# sns.pointplot(
#     data=df_spatialaverage, x='weather', y='avg_zeta', hue='mode',
#     palette={mode['name']: mode['color'] for mode in modes},

#     # dodge=1.0 - 1.0/2, # see https://seaborn.pydata.org/examples/jitter_stripplot.html
#     dodge=True,
#     ci=None
# )
# Plot violinplot
# sns.boxplot(
#         data=df_spatialaverage, x='weather', y='avg_zeta', hue='mode',
#         palette={mode['name']:mode['color'] for mode in modes}
#         )
sns.violinplot(
    data=df_spatialaverage[df_spatialaverage['weather'] == 'wet'],
    x='weather', y='avg_zeta', hue='blocks',
    palette={'no blocks': modes[1]['color'],
             'blocks': modes[3]['color']},
    dodge=True,
    alpha=0.5
)
sns.stripplot(
    data=df_spatialaverage[df_spatialaverage['weather'] == 'dry'],
    x='weather', y='avg_zeta', hue='blocks',
    palette={'no blocks': modes[0]['color'],
             'blocks': modes[2]['color']},
    dodge=True,
    alpha=0.5
)
plot_repeated_labels_only_once_in_legend()
plt.show()

# %% All params figure


def _get_piecewise_masks_in_zeta(zeta):
    positives_mask = 1*(zeta >= 0)
    negatives_mask = 1 - positives_mask
    return positives_mask, negatives_mask


def _mask_array_piecewise(arr, mask_1, mask_2):
    return arr*mask_1, arr*mask_2


def S(zeta, param):
    # in terms of zeta = h - dem
    s1, s2, t1, t2 = param
    positives_mask, negatives_mask = _get_piecewise_masks_in_zeta(zeta)
    zeta_pos, zeta_neg = _mask_array_piecewise(
        zeta, positives_mask, negatives_mask)
    sto_pos = 1
    sto_neg = s1*np.exp(s2*zeta)
    return sto_pos * positives_mask + sto_neg * negatives_mask


def T(zeta, param):
    # in terms of zeta = h - dem
    s1, s2, t1, t2 = param
    positives_mask = 1*(zeta > 0)
    negatives_mask = 1 - positives_mask
    zeta_pos, zeta_neg = _mask_array_piecewise(
        zeta, positives_mask, negatives_mask)

    t3 = t1/s1
    t4 = (t2-s2)/s1

    tra_pos = t3*np.exp(t4*zeta)
    tra_neg = t1*np.exp(t2*zeta)

    return tra_pos*positives_mask + tra_neg*negatives_mask


def K(zeta, param):
    # in terms of zeta = h - dem
    s1, s2, t1, t2 = param

    positives_mask = 1*(zeta > 0)
    negatives_mask = 1 - positives_mask
    zeta_pos, zeta_neg = _mask_array_piecewise(
        zeta, positives_mask, negatives_mask)

    t3 = t1/s1
    t4 = (t2-s2)/s1

    k_pos = t3*t4*np.exp(t4*zeta_pos)
    k_neg = t1*t2*np.exp(t2*zeta_neg)

    return k_pos*positives_mask + k_neg*negatives_mask

def D(zeta, param):
    return T(zeta, param)/S(zeta, param)

# Read params
# param_fn = parent_folder.joinpath('2d_calibration_parameters.xlsx')
# df_params = pd.read_excel(param_fn)
# New params
# p1 = [0.7, 0.9, 68.53, 18.97]
# p2 = [0.7, 0.9, 286.57, 4.54]
# p3 = [0.7, 0.9, 641.66, 2.03]
# Old params
p1 = [0.6, 0.5, 50, 2.5]
p2 = [0.6, 0.5, 50, 7.5]
p3 = [0.6, 0.5, 500, 2.5]
p4 = [0.6, 0.5, 500, 7.5]

zz = np.linspace(-1, 0.2, num=1000)

fig, axes = plt.subplots(nrows=2, ncols=3, sharey=True, figsize=(4.33*1.69/2*3, 4.33*1.69))

gs = axes[1,0].get_gridspec()
for ax in axes[1,:]: # remove axes in the bottom row
    ax.remove()
ax_D = fig.add_subplot(gs[1, :]) # create axis in top row spanning all the columns

# Set labels for subplots
for ax, label in zip(axes.flat, ["a)", "b)", "c)"]):
    ax.text(0 + 0.02, 1 - 0.02, s=label,
                transform=ax.transAxes,
                verticalalignment='top',
                fontsize=10,
                bbox=dict(facecolor='1.0', edgecolor='none', pad=3.0))
ax_D.text(0 + 0.02, 1 - 0.02, s='d)',
            transform=ax_D.transAxes,
            verticalalignment='top',
            fontsize=10,
            bbox=dict(facecolor='1.0', edgecolor='none', pad=3.0))
# storage
axes[0,0].grid(visible=True)
axes[0,0].set_xlabel(r'$S$')
axes[0,0].set_ylabel(r'$\zeta (m)$')
axes[0,0].plot(S(zz, p1), zz,
             color='black', linestyle='solid')

# Transmissivity
axes[0,1].grid(visible=True)
axes[0,1].set_xlabel(r'$T(m^2 d^{-1})$')
axes[0,1].set_xscale('log')

axes[0,1].plot(T(zz, p1), zz,
             color=cmap_parameterization(0), linestyle='solid',
             label='param 1')
axes[0,1].plot(T(zz, p2), zz,
             color=cmap_parameterization(1), linestyle='solid',
             label='param 2')
axes[0,1].plot(T(zz, p3), zz,
             color=cmap_parameterization(2), linestyle='solid',
             label='param 3')
axes[0,1].plot(T(zz, p4), zz,
             color=cmap_parameterization(3), linestyle='solid',
             label='param 4')

# axes[1].set_ylabel(r'$\zeta$')

# conductivity
axes[0,2].grid(visible=True)
axes[0,2].set_xlabel(r'$K(md^{-1})$')
axes[0,2].set_xscale('log')
axes[0,2].plot(K(zz, p1), zz,
             color=param_colors[0], linestyle='solid')
axes[0,2].plot(K(zz, p2), zz,
             color=param_colors[1], linestyle='solid')
axes[0,2].plot(K(zz, p3), zz,
             color=param_colors[2], linestyle='solid')
axes[0,2].plot(K(zz, p4), zz,
             color=param_colors[3], linestyle='solid')
# axes[2].set_ylabel(r'$\zeta$')

# Diffusivity
ax_D.set_xscale('log')
ax_D.grid(visible=True)
ax_D.set_xlabel(r'$D(m^{2}d^{-1})$')
ax_D.set_ylabel(r'$\zeta (m)$')
# ax_D.set_xscale('log')
ax_D.plot(D(zz, p1), zz,
             color=param_colors[0], linestyle='solid',
             label='param 1')
ax_D.plot(D(zz, p2), zz,
             color=param_colors[1], linestyle='solid',
             label='param 2')
ax_D.plot(D(zz, p3), zz,
             color=param_colors[2], linestyle='solid',
             label='param 3')
ax_D.plot(D(zz, p4), zz,
             color=param_colors[3], linestyle='solid',
             label='param 4')
# legend
ax_D.legend(loc='lower right')

fig.tight_layout()
plt.savefig(output_folder.joinpath(f'parameterization'), bbox_inches='tight')
plt.show()

# %% WTD rasters 
day = 10
parameter = 1
mode = 0 #nodry, nowet, yesdry, yeswet

plt.figure()
plt.title(f'day {day}, param {parameter}, {modes[mode]["name"]}')
cmap_wtd_raster.set_bad(color='white')
plt.axis('off')

plt.imshow(data_with_nan[parameter, mode, day, :, :],

        cmap=cmap_wtd_raster)
plt.colorbar()
fig.tight_layout()
plt.show()

# %% WTD Reality check different times
# TODO. In the future, this should take reality check WTDs
days = [10, 100, 200, 300]
parameter = 1
mode = 0 #nodry, nowet, yesdry, yeswet

cmap_wtd_raster.set_bad(color='white')

fig, axes = plt.subplots(nrows=2, ncols=2)
for ax, day in zip(axes.flat, days):
    ax.axis('off')
    ax.set_title(f'day {day}')

    im = ax.imshow(data_with_nan[parameter, mode, day, :, :],
            cmap=cmap_wtd_raster)

fig.colorbar(im, ax=axes.ravel().tolist())
plt.savefig(output_folder.joinpath(f'reality_check_wtd_several_days.png'), bbox_inches='tight')
plt.show()
# %% Precipitation & ET

fig, axes = plt.subplots(nrows=2, ncols=1,
        sharex=True, constrained_layout=True)
for ax, label in zip(axes.flat, ["a)", "b)"]):
    ax.text(0 + 0.02, 1 - 0.02, s=label,
                transform=ax.transAxes,
                verticalalignment='top',
                fontsize=10,
                bbox=dict(facecolor='1.0', edgecolor='none', pad=3.0))
ax_wet = axes[0]
ax_dry = axes[1]
ax_dry.set_xlabel('time (d)')
ax_dry.set_ylabel(r'$P - ET (mm) $')
ax_wet.set_ylabel(r'$P - ET (mm) $')
ax_wet_twin = ax_wet.twinx()
ax_wet_twin.set_ylabel('cumulative $P - ET$ (mm)')
ax_dry_twin = ax_dry.twinx()
ax_dry_twin.set_ylabel('cumulative $P - ET$ (mm)')

ax_left_limits = [min(wet_sourcesink) - 5, max(wet_sourcesink) + 5]
ax_right_limits = [min(np.cumsum(dry_sourcesink)) - 100, np.sum(wet_sourcesink) + 100]
ax_wet.set_ylim(ax_left_limits)
ax_wet_twin.set_ylim(ax_right_limits)
ax_dry.set_ylim(ax_left_limits)
ax_dry_twin.set_ylim(ax_right_limits)

ax_wet_twin.axhline(y=0, color='black', alpha=0.1)
ax_dry_twin.axhline(y=0, color='black', alpha=0.1)

ax_wet.bar(x=range(len(wet_sourcesink)),
        height=wet_sourcesink,
        label='wet year',
        alpha=0.4,
        color=modes[1]['color']
        )
ax_wet_twin.plot(range(len(wet_sourcesink)), np.cumsum(wet_sourcesink),
        color=modes[1]['color'])
ax_dry.bar(x=range(len(dry_sourcesink)),
        height=dry_sourcesink,
        label='dry year',
        alpha=0.4,
        color=modes[0]['color']
        )
ax_dry_twin.plot(range(len(dry_sourcesink)), np.cumsum(dry_sourcesink),
        color=modes[0]['color'])
ax_dry.legend(loc='upper center')
ax_wet.legend(loc='upper center')

plt.savefig(output_folder.joinpath(f'P_minus_ET'), bbox_inches='tight')
plt.show()




# %% Plot WTD diff vs distance to blocks for all params and weathers.
fig, axes = plt.subplots(nrows=N_PARAMS, ncols=2, sharex=True, sharey=True, figsize=(4.13, 4.13*2))
axes_wet = axes[:,0]
axes_dry = axes[:,1]
axes_wet[0].set_title('Wet')
axes_dry[0].set_title('Dry')
for ax in axes_wet:
    ax.set_ylabel(r'$ \Delta \bar{\zeta} (m)$')
for ax in axes[-1]:
    ax.set_xlabel('distance to block (m)')
for ax in axes.flat:
    ax.grid(visible='True', linewidth=0.5)
    ax.set_xlim([0, 1000])
    ax.tick_params(width=0)
    
for ax in axes[:,0]:
    ax.tick_params(axis='y', width=1)
for ax in axes[-1,:]:
    ax.tick_params(axis='x', width = 1)
axes[-1,0].set_xticks(axes[-1,0].get_xticks()[:-1]) # Remove some ticks from bottom axes
axes[-1,1].set_xticks(axes[-1,1].get_xticks()) # Remove some ticks from bottom axes

# for ax, label in zip(axes.flat, ["a)", "b)", "c)", "d)", "e)", "f)", "g)", "h)"]):
#     ax.text(1 - 0.15, 1 - 0.05, s=label,
#                 transform=ax.transAxes,
#                 verticalalignment='top',
#                 fontsize=10,
#                 bbox=dict(facecolor='1.0', edgecolor='none', pad=3.0))
    

for i_param, param_to_plot in enumerate(params_to_plot):
    # plot dry points
    axes_dry[i_param].plot(min_dist_array_to_blocks.ravel(), dry_diffs_temporal_avg[i_param, :, :].ravel(),
            label=f'param {param_to_plot}',
            color=param_colors[i_param],
            alpha=0.1,
            marker='.', markersize=3,
            linestyle='None')
    # plot dry mean line
    axes_dry[i_param].plot(bins, dry_mean_diff_binned[i_param],
            color='black')
    
    # plot wet points
    axes_wet[i_param].plot(min_dist_array_to_blocks.ravel(), wet_diffs_temporal_avg[i_param, :, :].ravel(),
            label=f'param {param_to_plot}',
            color=param_colors[i_param],
            alpha=0.1,
            marker='.', markersize=3,
            linestyle='None')
    axes_wet[i_param].plot(bins, wet_mean_diff_binned[i_param],
            color='black')

# for ax in axes.flat:   
#     leg = ax.legend()

# fig.tight_layout()
plt.subplots_adjust(wspace=0.0, hspace=0.0)
plt.savefig(output_folder.joinpath(f'spatial_extent_block_effect.png'),
            bbox_inches='tight')
plt.show()
#%% Plot some snapshots
MIN_DIFF = -1.0
MAX_DIFF = 1.0
cmap_diff.set_bad(color='white') # mask color


# snapshots info
params = [3, 3, 2, 2]; days = [360, 360, 360, 360] ; weathers = ['dry', 'wet', 'dry', 'wet']

for i in range(4):
    if weathers[i] == 'dry':
        diff_raster = dry_diffs_with_nan[params[i]-1, days[i]-1, :, :]
        title = f'Diff day {days[i]} dry period param {params[i]}'
        fname = f'diff_raster_dry_param{params[i]}_day{days[i]}.png'
    elif weathers[i] == 'wet':
        diff_raster = wet_diffs_with_nan[params[i]-1, days[i]-1, :, :]
        title = f'Diff day {days[i]} wet period param {params[i]}'
        fname = f'diff_raster_wet_param{params[i]}_day{days[i]}.png'

    plt.figure()
    plt.axis('off')
    plt.title(title, fontsize=10)
    im = plt.imshow(diff_raster, cmap=cmap_diff,
            vmin=MIN_DIFF, vmax=MAX_DIFF)
    cb = plt.colorbar(im, shrink=0.7)
    cb.ax.tick_params(labelsize=10)

    plt.savefig(output_folder.joinpath(fname),
                    bbox_inches='tight')


plt.show()
# %% Some results
print("Dry weather conditions resulted in a larger average rewetting compared to the wet conditions by a factor of:")
print(np.mean(dry_diffs_spatial_avg, axis=(0,1))/np.mean(wet_diffs_spatial_avg, axis=(0,1)))

print("The set of hydraulic peat properties that led to the maximum and minimum rewetting in wet conditions differed by a factor of:")
print(np.mean(wet_diffs_spatial_avg, axis=(1)).max()/np.mean(wet_diffs_spatial_avg, axis=(1)).min())

print("The set of hydraulic peat properties that led to the maximum and minimum rewetting in dry conditions differed by a factor of:")
print(np.mean(dry_diffs_spatial_avg, axis=(1)).max()/np.mean(dry_diffs_spatial_avg, axis=(1)).min())

print("Total annual sequestered CO2 Mg per ha for DRY parameters:")
print(cum_sequestered_dry_CO2_daily[:,-1])
print("Mean annual sequestered CO2 Mg per ha for DRY conditions:")
print(cum_sequestered_dry_CO2_daily[:,-1].mean())
print("Total annual sequestered CO2 Mg per ha for WET parameters:")
print(cum_sequestered_wet_CO2_daily[:,-1])
print("Mean annual sequestered CO2 Mg per ha for WET conditions:")
print(cum_sequestered_wet_CO2_daily[:,-1].mean())
# %%
