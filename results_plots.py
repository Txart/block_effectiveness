# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import rasterio
from pathlib import Path
from collections import OrderedDict
import seaborn as sns

# %% Define style
plt.style.use('seaborn-paper')
plt.rc('font', family='serif')
sns.set_context('paper', font_scale=1.5)

# %% Params
N_DAYS = 30
N_PARAMS = 2
params_to_plot = [1, 2]

# Output folder
output_folder = Path('output/plots')
# Data folder
# data_folder = Path('output')
data_folder = Path('hydro_for_plots')

# colormap
# see https://matplotlib.org/3.5.0/tutorials/colors/colormaps.html
cmap = cm.get_cmap('Dark2')

# Foldernames
modes = ([
    {'foldername': 'no_blocks_dry',
        'name': 'el Niño, no blocks',
        'color': cmap(1),
        'marker':'x',
        'linestyle':'dashed'},
    {'foldername': 'no_blocks_wet',
        'name': 'wet, no blocks',
        'color': cmap(0),
        'marker':'x',
        'linestyle':'dashed'},
    {'foldername': 'yes_blocks_dry',
        'name': 'el Niño, blocks',
        'color': cmap(1),
        'marker':'o',
        'linestyle':'solid'},
    {'foldername': 'yes_blocks_wet',
        'name': 'wet, blocks',
        'color': cmap(0),
        'marker':'o',
        'linestyle':'solid'},
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

#%% Read data
# Get raster dimensions from first raster file
template_rfn = data_folder.joinpath(
    f'params_number_1/{modes[0]["foldername"]}/zeta_after_1_DAYS.tif')
with rasterio.open(template_rfn, 'r') as src:
    template_raster = src.read(1)
raster_shape = template_raster.shape


def read_data(n_params, modes, n_days, rasters_shape):
    # data = [[[0]*N_DAYS for _ in range(len(mode_foldernames))] for _ in range(N_PARAMS)]
    data = np.zeros(shape=(n_params, len(modes), n_days,
                    rasters_shape[0], rasters_shape[1]))
    for n_param in range(N_PARAMS):
        param_folder = data_folder.joinpath(f'params_number_{int(n_param + 1)}')
        for n_mode, mode_foldername in enumerate([mode['foldername'] for mode in modes]):
            foldername = param_folder.joinpath(mode_foldername)
            for day in range(0, N_DAYS):
                fname = foldername.joinpath(
                    f'zeta_after_{int(day+1)}_DAYS.tif')
                with rasterio.open(fname, 'r') as src:
                    data[n_param][n_mode][day] = src.read(1)
    return data


# Array containing data mimicks folder structure
# Structure: [param_number, block_and_precip_type, day_number, raster]
data = read_data(N_PARAMS, modes, N_DAYS, raster_shape)

# Change raster nans to zeros
data = np.nan_to_num(data, nan=0.0)

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
alpha = 74.11/365 # Mg/ha/m/day
beta = 29.34/365 # Mg/ha/day
m_CO2_daily = -alpha * np.mean(data_capped_at_zero, axis=(3,4)) + beta # Mg/ha/day
cum_CO2_daily = np.cumsum(m_CO2_daily, axis=2)

avoided_wet_CO2_daily = m_CO2_daily[:, 3, :] - m_CO2_daily[:, 1, :] # blocked - noblocked
avoided_dry_CO2_daily = m_CO2_daily[:, 2, :] - m_CO2_daily[:, 0, :] 
cum_avoided_wet_CO2_daily = np.cumsum(avoided_wet_CO2_daily, axis=1)
cum_avoided_dry_CO2_daily = np.cumsum(avoided_dry_CO2_daily, axis=1)

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
co2_columns = ['day', 'avoided_CO2', 'cum_avoided_CO2', 'weather', 'param']
df_daily_dry_CO2 = pd.DataFrame(columns=co2_columns, index=range(N_DAYS * N_PARAMS))
df_daily_wet_CO2 = pd.DataFrame(columns=co2_columns, index=range(N_DAYS * N_PARAMS))

new_df = pd.DataFrame(columns=co2_columns, index=range(N_DAYS))

for i_param, param_name in enumerate(params_to_plot):
    # dry
    new_df['day'] = np.arange(1, N_DAYS+1)
    new_df['avoided_CO2'] = avoided_dry_CO2_daily[i_param] 
    new_df['cum_avoided_CO2'] = cum_avoided_dry_CO2_daily[i_param] 
    new_df['weather'] = 'dry'
    new_df['param'] = param_name
    df_daily_dry_CO2 = pd.concat([df_daily_dry_CO2, new_df], ignore_index=True)
    # wet
    new_df['day'] = np.arange(1, N_DAYS+1)
    new_df['avoided_CO2'] = avoided_wet_CO2_daily[i_param] 
    new_df['cum_avoided_CO2'] = cum_avoided_wet_CO2_daily[i_param] 
    new_df['weather'] = 'wet'
    new_df['param'] = param_name
    df_daily_wet_CO2 = pd.concat([df_daily_wet_CO2, new_df], ignore_index=True)

# %%------------------------------
# One param spatial_average vs time
# --------------------------------
for param_to_plot in params_to_plot:
    plt.figure()
    plt.title(f'parameter number {param_to_plot}')
    plt.xlabel('time (d)')
    plt.ylabel(r'$\bar{\langle\zeta\rangle} (m)$')
    for i_mode, mode in enumerate(modes):
        plt.plot(range(N_DAYS), spatial_average[int(param_to_plot - 1), i_mode],
                 label=mode['name'],
                 color=mode['color'],
                 linestyle=mode['linestyle'])
    # fill diff with color
    # dry year 
    plt.fill_between(x=range(N_DAYS),
                     y1=spatial_average[int(param_to_plot - 1), 2],
                     y2=spatial_average[int(param_to_plot - 1), 0],
                     color=modes[0]['color'],
                     alpha=0.1)
    # wet year
    plt.fill_between(x=range(N_DAYS),
                     y1=spatial_average[int(param_to_plot - 1), 3],
                     y2=spatial_average[int(param_to_plot - 1), 1],
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
    plt.plot(range(N_DAYS), cum_avoided_dry_CO2_daily[i_param], 
                 label='el Niño',
                 color=modes[0]['color']
                 )
    plt.plot(range(N_DAYS), cum_avoided_wet_CO2_daily[i_param], 
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
plt.figure()
plt.title('All parameters, wet, daily avg zeta over time')
p = sns.lineplot(
    data=df_daily_spatialaverage[df_daily_spatialaverage['weather'] == 'wet'],
    x='day', y='avg_zeta',
    style='blocks', style_order=(['blocks', 'no blocks']),
    ci='sd',
    color=modes[1]['color'],
)
p.set_xlabel('time (d)')
p.set_ylabel(r'$\bar{\langle\zeta\rangle} (m)$')
plot_repeated_labels_only_once_in_legend()
plt.savefig(output_folder.joinpath(f'avg_zeta_vs_time_WET_all_params'),
                bbox_inches='tight')
plt.show()

plt.figure()
plt.title('All parameters, dry, daily avg zeta over time')
p = sns.lineplot(
    data=df_daily_spatialaverage[df_daily_spatialaverage['weather'] == 'dry'],
    x='day', y='avg_zeta',
    style='blocks', style_order=(['blocks', 'no blocks']),
    color=modes[2]['color'],
    ci='sd')
p.set_xlabel('time (d)')
p.set_ylabel(r'$\bar{\langle\zeta\rangle} (m)$')
plot_repeated_labels_only_once_in_legend()
plt.savefig(output_folder.joinpath(f'avg_zeta_vs_time_DRY_all_params'),
                bbox_inches='tight')
plt.show()


# -----------------------------------------
# One param cumulative saved CO2 over time
# -----------------------------------------
plt.figure()
plt.title(f'All parameters, cumulative CO_2 emissions')
plt.xlabel('time (d)')
plt.ylabel('cum CO2 emissions $(Mg ha^{-1})$')
p = sns.lineplot(
        data=df_daily_dry_CO2,
        x='day', y='cum_avoided_CO2',
        label='dry',
        color=modes[2]['color'],
        ci='sd'
        )
p = sns.lineplot(
        data=df_daily_wet_CO2,
        x='day', y='cum_avoided_CO2',
        label='wet',
        color=modes[3]['color'],
        ci='sd'
        )
plt.legend()
plt.savefig(output_folder.joinpath(f'daily_cumulative_CO2_all_params'),
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
# TODO: REPLACE WITH TRUE FUNCS AND VALUES
# param_fn = '/home/txart/Programming/github/paper2/2d_calibration_parameters.xlsx'
# df_params = pd.read_excel(param_fn)
df_params = pd.DataFrame(columns=['s1', 's2', 't1', 't2'], index=range(8))
df_params.loc[:] = np.random.rand(8,4)
unique_s1_values = df

def sto(zeta, s1, s2):
    return s1*zeta**s2

zz = np.linspace(-2, 0.3, num=1000)


fig, axes = plt.subplots(nrows=1, ncols=3, sharey=True)
# storage 
axes[0].set_xlabel(r'$S$')
axes[0].set_ylabel(r'$\zeta (m)$')
axes[0].plot(sto())

# Transmissivity 
axes[1].set_xlabel(r'$T(m^2 d^{-1})$')
# axes[1].set_ylabel(r'$\zeta$')

# conductivity 
axes[2].set_xlabel(r'$K(md^{-1})$')
# axes[2].set_ylabel(r'$\zeta$')

plt.show()
