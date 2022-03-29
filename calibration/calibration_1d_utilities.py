#%%
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

#%% THE DIFFERENCE BETWEEN P AND TN SENSORS IS THAT THE LATTER HAVE MORE THAN ONE SENSOR IN THE TRANSECT.
# THAT'S WHY WE HAVE SEPARATE FUNCTIONS FOR EACH

def organize_data_into_transects_Psensors(relevant_transects, dfs_by_sensor):
    # From sensors to transects. Only for P sensors
    dfs_by_transect = {}
    for tname in relevant_transects:
        canal_wtd = pd.DataFrame(dfs_by_sensor[tname+'X']['WTD'].rename('WTD_canal'))
        inland_wtd = pd.DataFrame(dfs_by_sensor[tname]['WTD'].rename('WTD_inland'))
        # It seems that there are errors with multiple values for same julian day
        # Not very important for the future, but drop them here
        inland_wtd = inland_wtd[~inland_wtd.index.duplicated()]
        canal_wtd = canal_wtd[~canal_wtd.index.duplicated()]
        
        dfs_by_transect[tname] = pd.concat([inland_wtd, canal_wtd], axis=1)
    
    return dfs_by_transect
   
def organize_data_into_transects_TNsensors(relevant_transects, dfs_by_sensor): 
    dfs_by_transect = {}
    for tname in relevant_transects:
        canal_wtd = pd.DataFrame(dfs_by_sensor[tname+'1']['WTD'].rename('WTD_canal'))
        # inland_wtd2 = pd.DataFrame(dfs_by_sensor[tname]['WTD'].rename('WTD_inland'))
        inland_wtd3 = pd.DataFrame(dfs_by_sensor[tname+'3']['WTD'].rename('WTD_inland_3'))
        # inland_wtd4 = pd.DataFrame(dfs_by_sensor[tname]['WTD'].rename('WTD_inland'))
        inland_wtd5 = pd.DataFrame(dfs_by_sensor[tname+'5']['WTD'].rename('WTD_inland_5'))
        
        dfs_by_transect[tname] = pd.concat([canal_wtd, inland_wtd3, inland_wtd5], axis=1)
        
    return dfs_by_transect

def add_weather_data(relevant_transects, daily_weather_df, dfs_by_transect):
    all_data_by_transect = {}
    for tname in relevant_transects:
        source = pd.DataFrame((daily_weather_df['P'] - daily_weather_df['ET']).rename('p_minus_et'))
        # all_data_by_transect[tname] = pd.concat([dfs_by_transect[tname], source], axis=1, join='inner')
        all_data_by_transect[tname] = dfs_by_transect[tname].join(source)
    
    return all_data_by_transect

def slice_by_julian_days(all_data_by_transect, jday_bounds):
    dfs_sliced_relevant_transects = {}
    for key, df in all_data_by_transect.items():
        sliced_df = df.loc[jday_bounds[key][0]:jday_bounds[key][1]]
        dfs_sliced_relevant_transects[key] = sliced_df
    
    return dfs_sliced_relevant_transects

def get_transect_physical_information_Psensors(fn_transect_info):
    df_tra_info =  pd.read_excel(fn_transect_info)
    transect_elevations = {tname: df_tra_info[df_tra_info['Transect Name']==tname]['elevation_difference_canal_minus_inland(m)'].values[0] for tname in list(df_tra_info['Transect Name'].values) if 'P' in tname}
    transect_length = {tname: df_tra_info[df_tra_info['Transect Name']==tname]['distance_between_sensors(m)'].values[0] for tname in list(df_tra_info['Transect Name'].values) if 'P' in tname}

    return transect_elevations, transect_length

def get_transect_physical_information_TNsensors(fn_transect_info):
    df_tra_info =  pd.read_excel(fn_transect_info)
    sensor_elevations = {sname: df_tra_info[df_tra_info['Transect Name']==sname]['elevation_difference_canal_minus_inland(m)'].values[0] for sname in list(df_tra_info['Transect Name'].values) if 'TN' in sname}
    sensor_distance = {sname: df_tra_info[df_tra_info['Transect Name']==sname]['distance_between_sensors(m)'].values[0] for sname in list(df_tra_info['Transect Name'].values) if 'TN' in sname}

    return sensor_elevations, sensor_distance

def reorder_all_data_in_metadictionary_Psensors(dfs_sliced_relevant_transects, relevant_transects, fn_transect_info):
    transect_elevations, transect_length = get_transect_physical_information_Psensors(fn_transect_info)
    data_dict = {}
    for tran in relevant_transects:
        data_dict[tran] = {'df':dfs_sliced_relevant_transects[tran],
                        'transect_length': transect_length[tran],
                        'dem_at_canal_sensor': 10.0, # 10 is an arbitrary leveling number.
                        'dem_at_inland_sensor': 10.0 + transect_elevations[tran]}
    
    return data_dict

def reorder_all_data_in_metadictionary_TNsensors(dfs_sliced_relevant_transects, relevant_transects, fn_transect_info):
    sensor_elevations, sensor_distance = get_transect_physical_information_TNsensors(fn_transect_info)
    data_dict = {}
    for tran in relevant_transects:
        data_dict[tran] = {'df':dfs_sliced_relevant_transects[tran],
                       'transect_length': sensor_distance[tran+'5'],
                       'sensor_distances': [sensor_distance[sname] for sname in [tran+'2', tran+'3', tran+'4', tran+'5']], 
                       'dem_at_canal_sensor': 10.0, # 10 is an arbitrary leveling number.
                       'relative_height_at_inland_sensors': [sensor_elevations[sname] for sname in [tran+'2', tran+'3', tran+'4', tran+'5']]
                       }
    return data_dict

def get_dem_Psensors(data, nx):
    surface_height_at_canal_sensor = data['dem_at_canal_sensor']
    surface_height_at_inland_sensor = data['dem_at_inland_sensor']

    return np.linspace(surface_height_at_canal_sensor, surface_height_at_inland_sensor, num=nx)

def get_all_sensor_locations_TNsensors(data):
    return [0] + data['sensor_distances']

def get_dem_TNsensors(data, nx):
    rel_heights = [0] + data['relative_height_at_inland_sensors']
    dem_sensors = data['dem_at_canal_sensor'] + np.array(rel_heights)
    xs = get_all_sensor_locations_TNsensors(data)

    dem_interp_func = interp1d(x=xs, y=dem_sensors)
    return dem_interp_func(np.linspace(start=xs[0], stop=xs[-1], num=nx))

def get_sensor_measurements_Psensors(data):
    zeta_right_measurements = data['df']['WTD_inland'].to_numpy()
    ZETA_LEFT_DIRIS = data['df']['WTD_canal'].to_numpy()
    
    return ZETA_LEFT_DIRIS, zeta_right_measurements

def get_sensor_measurements_TNsensors(data):
    # Only sensors 1, 3 and 5 are automatic TN transects
    ZETA_LEFT_DIRIS = data['df']['WTD_canal'].to_numpy()
    zeta_middle_measurements = data['df']['WTD_inland_3'].to_numpy()
    ZETA_RIGHT_DIRIS = data['df']['WTD_inland_5'].to_numpy()
    
    return ZETA_LEFT_DIRIS, zeta_middle_measurements, ZETA_RIGHT_DIRIS



def compute_initial_zeta_TNsensors(ZETA_LEFT_DIRIS, zeta_middle_measurements, ZETA_RIGHT_DIRIS, data, nx):
    # Initial state of WTD for TNx sensors is just a linear interpolation through data points.
    zeta_left_ini = ZETA_LEFT_DIRIS[0]
    zeta_right_ini = ZETA_RIGHT_DIRIS[0]
    
    xs = [get_all_sensor_locations_TNsensors(data)[i] for i in [0,2,4]] # only 1, 3 and 5 sensors have measurements
    zs = [ZETA_LEFT_DIRIS[0], zeta_middle_measurements[0], ZETA_RIGHT_DIRIS[0]]
    zeta_interp_func = interp1d(x=xs, y=zs)
    zeta_ini = zeta_interp_func(np.linspace(start=xs[0], stop=xs[-1], num=nx))
    
    return zeta_ini

