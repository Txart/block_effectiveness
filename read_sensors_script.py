#%%
import pandas as pd
import numpy as np
from pathlib import Path
import rasterio
import pickle
from scipy.stats import zscore


#%% data location
sensor_data_parent_folder = Path('data/WTD_sensors')

#%% Read WTD transect dipwell data
# 2020
df_wtd_transects_20 = pd.read_csv(Path.joinpath(sensor_data_parent_folder, 'WTL_PatPos_2020.csv'), sep=',')
df_wtd_transects_canal_20 = pd.read_csv(Path.joinpath(sensor_data_parent_folder, 'WTL_PatPos_Canal_2020.csv'), sep=',')
df_wtd_other_20 = pd.read_csv(Path.joinpath(sensor_data_parent_folder, 'WTL_Other_2020.csv'), sep=',')

# 2021
df_wtd_transects_21 = pd.read_csv(Path.joinpath(sensor_data_parent_folder, 'WTL_PatPos_2021.csv'), sep=';')
df_wtd_transects_canal_21 = pd.read_csv(Path.joinpath(sensor_data_parent_folder, 'WaterLevel_Adjusted_2021.csv'), sep=';')
df_wtd_other_21 = pd.read_csv(Path.joinpath(sensor_data_parent_folder, 'WTL_Other_2021.csv'), sep=';')

# groups of dfs
all_dfs_2020 = [df_wtd_transects_20, df_wtd_transects_canal_20, df_wtd_other_20]
all_dfs_2021 =[df_wtd_transects_21, df_wtd_transects_canal_21, df_wtd_other_21]

all_dfs = all_dfs_2020 + all_dfs_2021

# 2021 data goes to index 366 -> 730
for df in all_dfs_2021:
    df.index = df.index + 366
    
#%% combine dfs
df_2020 = pd.concat(all_dfs_2020, axis=1)
df_2021 = pd.concat(all_dfs_2021, axis=1)

# Get sensors that were present in the 2 years to be able to concatenate the dfs
snames_2020 = {n for n in df_2020.columns}
snames_2021 = {n for n in df_2021.columns}
universal_snames = list(snames_2020.intersection(snames_2021))

df_all = pd.concat([df_2020[universal_snames], df_2021[universal_snames]], axis=0)

# Eliminate Date column
df_all = df_all.drop(['Date'], axis=1)

# Eliminate all-NaN columns
df_all = df_all.dropna(axis=1, how='all')

# Convert to floats
df_all = df_all.astype(float)

#%% Convert to meters
df_all = df_all *0.01

#%% Remove outliers

def remove_outliers(df, STDs):
    # values further away than STDs amount of standard deviations will be eliminated
    for colname, values in df.iteritems():
        no_nan_indices = values.dropna().index
        zscore_more_than_stds = (np.abs(zscore(values.dropna())) > STDs)
        # If value above x STDs, set it to NaN
        for i, index in enumerate(no_nan_indices):
            if zscore_more_than_stds[i]:
                df.loc[index,colname] = np.nan
    return df

df_all = remove_outliers(df=df_all, STDs=3)

# %% Plot to check
df_all.plot(legend=False)

#%% Get all sensor coordinates
# All sensors
df_dipwell_coords = pd.read_csv(Path.joinpath(sensor_data_parent_folder, 'all_dipwell_coords_projected.csv'))
dipwell_coords = {}
for _, row in df_dipwell_coords.iterrows():
    dipwell_coords[row.ID] = (row.x, row.y)
    
#%% Create initial sensor pickle


def sample_raster_from_list_of_points(raster_filename:str, point_coords:np.ndarray) ->np.ndarray:
    with rasterio.open(raster_filename) as src:
        return np.array([value[0] for value in src.sample(point_coords)])

# # Read initial dipwell data
FIRST_DAY = 22 # 22 seems to be a good day with about 30 sensors on. Before that, less than 8
no_nan_initial_sensor_data = df_all.iloc[FIRST_DAY][~df_all.iloc[FIRST_DAY].isnull()]
no_nan_initial_sensor_data = no_nan_initial_sensor_data.drop(['BU', 'SE', 'SI', 'LE']) # there's no location info about these
no_nan_initial_sensor_names = no_nan_initial_sensor_data.index
no_nan_initial_sensor_measurements = np.array(no_nan_initial_sensor_data)

initial_sensor_coords = [dipwell_coords[sn] for sn in no_nan_initial_sensor_names]

fn_sensor_pickle = "initial_sensor_pickle.p"
pickle.dump([initial_sensor_coords, no_nan_initial_sensor_measurements],
            file=open(fn_sensor_pickle, 'wb'))

# %% Get first day sensor measurements.
