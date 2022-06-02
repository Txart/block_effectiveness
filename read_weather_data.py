# %%
from pathlib import Path
import pandas as pd
import numpy as np


# %%
def _year_type_to_file_name(year_type: str) -> str:
    data_folder = Path('data/Raw csv')

    if year_type == 'elnino':
        filename = data_folder.joinpath('SultanThaha_1997_Precip.csv')
    elif year_type == 'normal':
        filename = data_folder.joinpath('SultanThaha_2013_Precip.csv')
    else:
        raise ValueError(
            'Unrecognized year type. Possible values: "elnino", "lanina"')

    return filename

def _fill_nodatas_with_mean(precip:np.ndarray) -> np.ndarray:
    nodata_points = np.isclose(precip, 8888) # 8888 is nodata value    
    precip_without_nodatas = precip.copy() 
    precip_without_nodatas[nodata_points] = 0
    mean_value = precip_without_nodatas.mean()
    
    # change nodata with mean
    precip[nodata_points] = 0
    return precip

def get_sultan_thaha_precip(year_type: str) -> np.ndarray:
    filename = _year_type_to_file_name(year_type)
    df = pd.read_csv(filename, sep=',')

    precip_array = df['RR'].to_numpy()
    precip_array = _fill_nodatas_with_mean(precip_array)

    # Catch nodatas and fill with mean
    precip_array = _fill_nodatas_with_mean(precip_array)

    precip_array = precip_array/1000  # mm/day -> m/day

    return precip_array

def constant_ET():
    # Hirano 2015 daily mean ET 4.17 mm
    return 4.17/1000

def get_daily_net_source(year_type:str) ->np.ndarray:
    return get_sultan_thaha_precip(year_type=year_type) - constant_ET()
# %%
