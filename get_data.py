#%%
import pandas as pd
import numpy as np
from pathlib import Path
import platform
import sys

#%%

def fill_weather_dataframe_missing_values_with_reference_patrol_post(dataframe, patrol_post_names, reference_patrol_post_name):
    # Fill  missing values with reference station's recordings
    for name in set(patrol_post_names) - set([reference_patrol_post_name]):
        for day in range(365):
            if np.isnan(dataframe.loc[day, name]):
                dataframe.at[day, name] = dataframe.loc[day, reference_patrol_post_name]

def get_P_minus_ET_dataframe(data_parent_folder):
    if platform.system() == 'Linux':
        sys.path.insert(1, r'~/Programming/GitHub/blopti_dev') # insert at 1, 0 is the script path
    elif platform.system() == 'Windows':  
        sys.path.insert(1, r'C:\Users\03125327\github\blopti_dev') # insert at 1, 0 is the script path
    
    import evapotranspiration_fao
    import estimate_daily_et
    
    patrol_post_names = ['Buring', 'Lestari', 'MainCamp', 'Ramin', 'Serindit', 'Sialang']
        
    df_rain = pd.read_csv(Path.joinpath(data_parent_folder, 'Rain2021.csv'), sep=';')
    df_relhum = pd.read_csv(Path.joinpath(data_parent_folder, 'RelHum2021.csv'), sep=';')
    df_maxtemp = pd.read_csv(Path.joinpath(data_parent_folder, 'Temp_Max2021.csv'), sep=';')
    df_mintemp = pd.read_csv(Path.joinpath(data_parent_folder, 'Temp_Min2021.csv'), sep=';')
    df_avgtemp = pd.read_csv(Path.joinpath(data_parent_folder, 'Temp_Average2021.csv'), sep=';')
    df_windspeed = pd.read_csv(Path.joinpath(data_parent_folder, 'WindSpeed2021.csv'), sep=';')
    
    # rename "Main Camp" to "MainCamp"
    for df in [df_rain, df_relhum, df_maxtemp, df_mintemp, df_avgtemp, df_windspeed]:
        df.rename(columns={'Main Camp': 'MainCamp'}, inplace=True)

    # Change units in rain from mm/day to m/day
    df_rain[patrol_post_names] = df_rain[patrol_post_names] * 0.001
    
    # Fill missing values
    # Sialang has all rainfall and relhum values
    fill_weather_dataframe_missing_values_with_reference_patrol_post(df_rain, patrol_post_names, reference_patrol_post_name='Sialang')
    fill_weather_dataframe_missing_values_with_reference_patrol_post(df_relhum,  patrol_post_names, reference_patrol_post_name='Sialang')

    # Compute ET
    jday_to_month = pd.to_datetime(df_relhum.Date).dt.month
  
    # Compute air pressure in the station given sea level air pressure
    air_pressure = evapotranspiration_fao.pressure_from_altitude(8.0) 

    df_ET = pd.DataFrame()
    for patrol_name in patrol_post_names:
        daily_ET = [0]*365
        
        for day in range(0, 365):
            n_month = jday_to_month[day]
            Tmax = df_maxtemp.loc[n_month-1, patrol_name]
            Tmin = df_mintemp.loc[n_month-1, patrol_name]
            Tmean = df_avgtemp.loc[n_month-1, patrol_name]
            relhum = df_relhum.loc[day, patrol_name]
            windspeed = df_windspeed.loc[day, 'Average Windspeed (m/s)']
            
            ET, _, _, _, _ = np.array(estimate_daily_et.compute_ET(day, Tmax=Tmax, Tave=Tmean, Tmin=Tmin,
                                                                RH=relhum, air_pressure=air_pressure, U=windspeed,
                                                                Kc=0.8, KRS=0.16))
            daily_ET[day] = ET

        df_ET[patrol_name] = np.array(daily_ET) * 0.001 # m/day
        
    df_p_minus_et = df_rain - df_ET
    df_p_minus_et.drop('Date', axis='columns', inplace=True)
    return df_p_minus_et
        