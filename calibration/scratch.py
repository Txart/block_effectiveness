#%%
import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from pathlib import Path
import pandas as pd
from multiprocessing import Pool
import emcee
import argparse
import platform
from scipy.interpolate import interp1d
from scipy.spatial import distance
import geopandas as gpd
import pickle
from tqdm import tqdm
import rasterio

#%% platform fork
if platform.system() == 'Linux': # Working on CSC machines
    raise NotImplementedError()

elif platform.system() == 'Windows':    
    data_parent_folder = Path(r"C:\Users\03125327\Dropbox\PhD\Computation\ForestCarbon\2021 SMPP WTD customer work\0. Raw Data")
    parent_directory = Path(r"C:\Users\03125327\github\blopti_dev")
    
    # import from blopti_dev
    sys.path.insert(1, r'C:\Users\03125327\github\blopti_dev') # insert at 1, 0 is the script path
    import estimate_daily_et
    import evapotranspiration_fao

#%% General stuff
patrol_post_names = ['Buring', 'Lestari', 'MainCamp', 'Ramin', 'Serindit', 'Sialang']

# %% Read weather data

df_rain = pd.read_csv(Path.joinpath(data_parent_folder, 'Raw csv/Rain2021.csv'), sep=';')
df_relhum = pd.read_csv(Path.joinpath(data_parent_folder, 'Raw csv/RelHum2021.csv'), sep=';')
df_maxtemp = pd.read_csv(Path.joinpath(data_parent_folder, 'Raw csv/Temp_Max2021.csv'), sep=';')
df_mintemp = pd.read_csv(Path.joinpath(data_parent_folder, 'Raw csv/Temp_Min2021.csv'), sep=';')
df_avgtemp = pd.read_csv(Path.joinpath(data_parent_folder, 'Raw csv/Temp_Average2021.csv'), sep=';')
df_windspeed = pd.read_csv(Path.joinpath(data_parent_folder, 'Raw csv/WindSpeed2021.csv'), sep=';')

print ('\n missing values')
print(df_maxtemp.isnull().sum())

# rename "Main Camp" to "MainCamp"
for df in [df_rain, df_relhum, df_maxtemp, df_mintemp, df_avgtemp, df_windspeed]:
    df.rename(columns={'Main Camp': 'MainCamp'}, inplace=True)

# Change units in rain from mm/day to m/day
df_rain[patrol_post_names] = df_rain[patrol_post_names] * 0.001


# %% Fill in missing values
def fill_weather_dataframe_missing_values_with_reference_patrol_post(dataframe, reference_patrol_post_name):
    # Fill  missing values with reference station's recordings
    for name in set(patrol_post_names) - set([reference_patrol_post_name]):
        for day in range(365):
            if np.isnan(dataframe.loc[day, name]):
                dataframe.at[day, name] = dataframe.loc[day, reference_patrol_post_name]

# Sialang has all rainfall and relhum values
fill_weather_dataframe_missing_values_with_reference_patrol_post(df_rain, reference_patrol_post_name='Sialang')
fill_weather_dataframe_missing_values_with_reference_patrol_post(df_relhum, reference_patrol_post_name='Sialang')

# %% Compute ET
jday_to_month = pd.to_datetime(df_relhum.Date).dt.month
  
# Compute air pressure in the station given sea level air pressure
air_pressure = evapotranspiration_fao.pressure_from_altitude(8.0) 

df_ET = pd.DataFrame()
for patrol_name in patrol_post_names:
    daily_ET = [0]*365
    
    for day in range(0, 365): # it was a leap year!
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

# %% Source = P - ET
df_source = df_rain - df_ET

#%% Read WTD transect dipwell data
df_wtd_transects = pd.read_csv(Path.joinpath(data_parent_folder, 'Raw csv/WTL_PatPos_2021.csv'), sep=';')
df_wtd_transects_canal = pd.read_csv(Path.joinpath(data_parent_folder, 'Raw csv/WaterLevel_Adjusted_2021.csv'), sep=';')


# Make dict of dataframes separated by transect
dict_of_transects = {}
small_names = ['MC', 'SI', 'SE', 'LE', 'RA']
for small_name in small_names:
    df_wtd_transects[small_name + '.0'] = df_wtd_transects_canal[small_name]
    all_col_names_with_name = [name for name in df_wtd_transects.columns if small_name in name]
    dict_of_transects[small_name] = df_wtd_transects[all_col_names_with_name]
   
#%% Visualize WTD transect data
for name, df in dict_of_transects.items():
    df.plot() 
    
# %% Transect sensor info
from scipy.interpolate import interp1d
NX = 50

# Elevation Data from DIpwell Elevation Report .xslx
dipwell_separation = [5, 25, 50, 100, 150] # metres
dipwell_heights = {'MC': [0.8, 1.21, 1.01, 1.09, 0.93],
                  'SI': [1.19, 0.9, 1.19, 1.1, 1.07],
                  'RA': [1.7, 1.58, 0.99, 1.02, 1.45],
                  'SE': [0.76, 1.07, 0.93, 1.21, 1.21],
                  'LE': [0.67, 0.58, 0.76, 1.06, 1.11]}

transect_dem = {}
for name, heights in dipwell_heights.items():
    inter_func = interp1d(x=dipwell_separation, y=heights)
    xx = np.linspace(start=5, stop=150, num=NX)
    transect_dem[name] = inter_func(xx)

# %% Read other dipwell data
df_wtd_other = pd.read_csv(Path.joinpath(data_parent_folder, 'Raw csv/WTL_Other_2021.csv'), sep=';')


#%% Put all dipwell data into same dataframe
all_dfs = list(dict_of_transects.values()) + [df_wtd_other]
all_sensor_WTD = pd.concat(all_dfs, axis=1)

# remove Date column
all_sensor_WTD.drop(labels='Date', axis='columns', inplace=True)

# Units: cm -> m
all_sensor_WTD = all_sensor_WTD/100

#%% Remove outliers from dipwell measurements
from scipy.stats import zscore
for colname, values in all_sensor_WTD.iteritems():
    no_nan_indices = values.dropna().index
    zscore_more_than_3std = (np.abs(zscore(values.dropna())) > 3)
    # If value above 3 STD, set it to NaN
    for i, index in enumerate(no_nan_indices):
        if zscore_more_than_3std[i]:
            all_sensor_WTD.loc[index,colname] = np.nan

# Plot
all_sensor_WTD.plot(legend=False, marker='.', linestyle='None')

#%% Get all sensor coordinates
# All sensors
df_dipwell_loc = pd.read_csv(Path.joinpath(data_parent_folder, 'Raw csv/all_dipwell_coords_projected.csv'))
dipwell_coords = {}
for _, row in df_dipwell_loc.iterrows():
    dipwell_coords[row.ID] = (row.x, row.y)

# UNCOMMENT IF WANTING TO USE CANAL DIPWELL DATA AS WELL
# Canal sensors assumed to have same coords as 1st sensors in the transect (in reality they are 5 m away)
for tname in small_names:
    dipwell_coords[tname + '.0'] = dipwell_coords[tname + '.1']
    

# %% Get best initial raster
# Compare measured and modelled
import rasterio
def sample_raster_from_list_of_points(raster_filename:str, point_coords:np.ndarray) ->np.ndarray:
    with rasterio.open(raster_filename) as src:
        return np.array([value[0] for value in src.sample(point_coords)])
        
# def sample_raster_from_list_of_points(raster_filename:str, point_coords:np.ndarray) ->np.ndarray:
#     raster_src = rasterio.open(raster_filename)
#     return np.array([raster_value[0] for raster_value in raster_src.sample(point_coords)])
 
out_raster_folder_path = parent_directory.joinpath('ForestCarbon2022\WTDrasters_for_ini_cond')
N_RASTERS = 0

# Read initial dipwell data
# Remove canal sensors: they are very close to the first dipwell and have peculiar values (and peculiar physics)

for tname in small_names:
    canal_dipwell_name = tname + '.0'
    if canal_dipwell_name in dipwell_coords.keys():
        dipwell_coords.pop(canal_dipwell_name)
    if canal_dipwell_name in all_sensor_WTD.columns:
        all_sensor_WTD.drop(labels=canal_dipwell_name, axis='columns', inplace=True)

    
no_nan_initial_sensor_data = all_sensor_WTD.iloc[0][~all_sensor_WTD.iloc[0].isnull()]
no_nan_initial_sensor_names = no_nan_initial_sensor_data.index
no_nan_initial_sensor_measurements = np.array(no_nan_initial_sensor_data)

initial_sensor_coords = [dipwell_coords[sn] for sn in no_nan_initial_sensor_names]

fitness = [0]*N_RASTERS

for day in range(N_RASTERS):
    #get raster values at sensor positions
    fn_dem_raster = Path.joinpath(out_raster_folder_path, f"t1-5_t2-1_n1-5_n2-1_zeta_after_{day}DAYS.tif")
    raster_values_at_sensor_locations = sample_raster_from_list_of_points(raster_filename=fn_dem_raster, point_coords=initial_sensor_coords)
    # Compute fitness
    fitness[day] = np.linalg.norm(raster_values_at_sensor_locations - no_nan_initial_sensor_measurements)

 
fitness = np.array(fitness)
best_raster = np.argwhere(fitness == fitness.min())[0]   
# VIsualize fitness
plt.figure()
plt.xlabel('raster number (DAYS)')
plt.ylabel('norm (measured - simulated)')
plt.plot(fitness)
plt.axvline(x=best_raster, color = 'red', label='best raster for ini cond.')
plt.legend()

    
# %% Read best ini cond
best_ic_fn = Path.joinpath(out_raster_folder_path, f"t1-5_t2-1_n1-5_n2-1_zeta_after_{int(best_raster)}DAYS.tif")
best_ini_zeta = preprocess_data.read_raster(best_ic_fn)


#%% Pickle dump initial sensor locations and values
fn_sensor_pickle = parent_directory.joinpath("ForestCarbon2022/sensor_pickle.p")
pickle.dump([initial_sensor_coords, no_nan_initial_sensor_measurements],
            file=open(fn_sensor_pickle, 'wb'))

# %% Compare all measured and modelled
def get_mesh_centroids_closest_to_sensor_locations(mesh, sensor_coords):
    meshcenters = mesh.cellCenters.value.T
    dists = distance.cdist(sensor_coords, meshcenters)
    mesh_number_corresponding_to_sensor_coords = np.argmin(dists, axis=1)
    return mesh_number_corresponding_to_sensor_coords

# Drop WTD sensor columns from dataframe whose position is unknown 
sensor_names_with_unknown_coords = [sname for sname in list(all_sensor_WTD.columns) if sname not in dipwell_coords]
all_sensor_WTD.drop(sensor_names_with_unknown_coords, axis='columns', inplace=True)

# Read mesh
import fipy as fp
mesh_fn = r"C:\Users\03125327\Dropbox\PhD\Computation\ForestCarbon\2021 SMPP WTD customer work\qgis_derivated_data\mesh\mesh_0.05.msh2" # triangular mesh
mesh = fp.Gmsh2D(mesh_fn)

# To find zeta mesh value at all sensor points, use sensor_name_to_pos_in_mesh
sensor_names = list(all_sensor_WTD.columns)
sensor_coords = [dipwell_coords[sn] for sn in sensor_names]

# The same but with mesh instead of raster
# sensor_position_in_mesh = get_mesh_centroids_closest_to_sensor_locations(mesh, sensor_coords)
# sensor_name_to_pos_in_mesh = {sensor_names[i]: sensor_position_in_mesh[i] for i in range(len(sensor_coords))}

# Get all modelled values at sensors in a big dict of dataframes
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


yesblocks_directory = parent_directory.joinpath(f"ForestCarbon2022/2d_calibration_output/params_number_0/yes_blocks_diriBC_full_diriBCexterior")
noblocks_directory = parent_directory.joinpath(f"ForestCarbon2022/2d_calibration_output/params_number_0/no_blocks_diriBC_full_diriBCexterior")

ndays = 365
modeled_yesblocks = create_modelled_result_dataframe(days_list=range(1, ndays+1), folder_path=yesblocks_directory)
modeled_noblocks = create_modelled_result_dataframe(days_list=range(1, ndays+1), folder_path=noblocks_directory)
    
# %% Plot measured and modelled

def S(zeta, s1, s2):
    # in terms of zeta = h - dem
    return s1*np.exp(s2*zeta)

def T(zeta, b, t1, t2):
    # in terms of zeta = h - dem
    return t1 * (np.exp(t2*zeta) - np.exp(-t2*b))

def variable_n_manning(y: np.ndarray, y_porous_threshold: np.ndarray, n1: float, n2: float) -> np.ndarray:
    # Change of variables to simplify coding. zeta is a local variable and has nothing to do with zeta at peatland hydrology.
    zeta = y - y_porous_threshold
    # Cap at porous threshold
    zeta[zeta < 0] = 0.0
    N_POROUS_THRESHOLD = 10000 # value at which n is cappe
    n_manning = N_POROUS_THRESHOLD / np.exp(n1 * zeta**n2)
    return n_manning

# Get params values
params_fn = Path.joinpath(parent_directory, 'ForestCarbon2022/2d_calibration_parameters.xlsx')
PARAMS = pd.read_excel(params_fn)

# Measured
all_sensor_WTD.plot(legend=False, marker='.', linestyle='None', title='measured')

# Modelled
NPARAMS = 1
for n_param in range(NPARAMS):
    PLOTCOLS = 3
    PLOTROWS = 3
    fig=plt.figure(figsize=(20,20), dpi=300)
    ax1=fig.add_subplot(PLOTROWS,PLOTCOLS,(1,3)) # extend plot to axes 1 to 3
    ax2 = fig.add_subplot(PLOTROWS,PLOTCOLS,(4,6))
    axT=fig.add_subplot(PLOTROWS,PLOTCOLS,7)
    axS=fig.add_subplot(PLOTROWS,PLOTCOLS,8)
    axn=fig.add_subplot(PLOTROWS,PLOTCOLS,9)
    
    # Plot measured
    all_sensor_WTD.plot(legend=False, marker='.', linestyle='None', title='measured', ax=ax1, alpha= 0.1, color='grey')
    all_sensor_WTD.plot(legend=False, marker='.', linestyle='None', title='measured', ax=ax2, alpha= 0.1, color='grey')
    
    # Plot modelled
    modeled_yesblocks.plot(ax=ax1, legend=False, title=f'modelled with blocks, param number {n_param}')
    modeled_noblocks.plot(ax=ax2, legend=False, title=f'modelled without blocks, param number {n_param}')
    
    # Plot physical properties
    s1 = PARAMS.loc[n_param, 's1']
    s2 = PARAMS.loc[n_param, 's2']
    t1 = PARAMS.loc[n_param, 't1']
    t2 = PARAMS.loc[n_param, 't2']
    porous_threshold_below_dem = PARAMS.loc[n_param, 'porous_threshold']
    n1 = PARAMS.loc[n_param, 'n1']
    n2 = PARAMS.loc[n_param, 'n2']
    # Plot T
    zz = np.linspace(start=-8, stop=0.5, num=100)
    axT.plot(T(zz, 8, t1, t2), zz)
    axT.set_title('Transmissivity')
    axT.set_ylabel('$\zeta (m)$')
    axT.set_xlabel('T')
    # Plot S
    axS.plot(S(zz, s1, s2), zz)
    axS.set_title('Specific yield')
    axS.set_ylabel('$\zeta (m)$')
    axS.set_xlabel('S')
    # Plot n
    dem = 8
    yy = dem + zz
    axn.plot(yy, variable_n_manning(yy, dem - porous_threshold_below_dem, n1, n2))
    axn.axvline(x=dem, ymin=0, ymax=10000, label='surface', color='brown')
    axn.set_title('n_manning')
    axn.set_xlabel('$h (m)$')
    axn.set_ylabel('n_manning')
    plt.legend()
    
    plt.savefig(fname=parent_directory.joinpath(f"ForestCarbon2022/2d_calibration_output/PLOTS/param_{n_param}"))


# %% Plot several points in the landscape over the year
# Get data
ndays = 365
point_coords = [(408858,9778629), (404494,9778629)]
WTD_yesblocks_at_points = np.zeros(shape=(ndays, len(point_coords)))
WTD_noblocks_at_points = np.zeros(shape=(ndays, len(point_coords)))

for day in range(1, ndays+1):
    for point in point_coords:
        zeta_yesblock_fn = Path.joinpath(yesblocks_directory, f"zeta_after_{int(day)}_DAYS.tif")
        WTD_yesblocks_at_points[day-1] = sample_raster_from_list_of_points(raster_filename=zeta_yesblock_fn, point_coords=point_coords)
        zeta_noblock_fn = Path.joinpath(noblocks_directory, f"zeta_after_{int(day)}_DAYS.tif")
        WTD_noblocks_at_points[day-1] = sample_raster_from_list_of_points(raster_filename=zeta_noblock_fn, point_coords=point_coords)

#%% Plot
for i, point in enumerate(point_coords):
    plt.figure()
    plt.plot(WTD_yesblocks_at_points[:,i], label=f'point {point}')

# %% Plot some individual sensors
sensor = 'MC.3'
modeled_yesblocks[sensor].plot(legend=True)
all_sensor_WTD[sensor].plot(legend=True, marker='.', linestyle='None')

# %% Choose best-fitting parameters

norms = {}

measured_flat = all_sensor_WTD.values.ravel()
for n_param in range(NPARAMS):
    modeled_flat = all_modelled[n_param].values.ravel()

    # mask NaNs
    nan_mask = ~ np.isnan(measured_flat)

    # Norm of the difference
    norm = np.linalg.norm(measured_flat[nan_mask] - modeled_flat[nan_mask])
    
    # Append to result
    norms[n_param] = norm

# Plot norms
plt.figure()
plt.plot([norms[npara] for npara in range(NPARAMS)]  , 'x')
plt.xlabel('PARAM NUMBER')
plt.ylabel('Norm of the difference')

# Compute best
best_parameters = np.array(list(norms.values())).argmin()
print(f"Best set of parameters are PARAMETERS NUMBER {best_parameters}")
# %% Output downstream nodes
import cwl_utilities
_, downstream_nodes = cwl_utilities.infer_extreme_nodes_with_networkx(channel_network.graph)

xcoords = []; ycoords = []
for n in downstream_nodes:
    xcoords.append(channel_network.graph.nodes[n]['x'])
    ycoords.append(channel_network.graph.nodes[n]['y'])
    
downstream_points_gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(xcoords, ycoords))
downstream_points_gdf.to_file(r"C:\Users\03125327\Desktop\DOWNSTREAM_POINTS_NEW.shp")




# %%
from fipy import CellVariable, Gmsh2D, TransientTerm, DiffusionTerm, Viewer
from fipy.tools import numerix

cellSize = 0.05
radius = 1.

mesh = Gmsh2D('''
              cellSize = %(cellSize)g;
              radius = %(radius)g;
              Point(1) = {0, 0, 0, cellSize};
              Point(2) = {-radius, 0, 0, cellSize};
              Point(3) = {0, radius, 0, cellSize};
              Point(4) = {radius, 0, 0, cellSize};
              Point(5) = {0, -radius, 0, cellSize};
              Circle(6) = {2, 1, 3};
              Circle(7) = {3, 1, 4};
              Circle(8) = {4, 1, 5};
              Circle(9) = {5, 1, 2};
              Line Loop(10) = {6, 7, 8, 9};
              Plane Surface(11) = {10};
              ''' % locals())

phi = CellVariable(name = "solution variable",
                   mesh = mesh,
                   value = 0.)
d = CellVariable(name = "d", mesh = mesh, value = 2.)

viewer = None

if __name__ == '__main__':
    try:
        viewer = Viewer(vars=phi, datamin=-3, datamax=3.)
        viewer.plotMesh()
    except:
        print("Unable to create a viewer for an irregular mesh (try Matplotlib2DViewer or MayaviViewer)")
        
D = 1.
eq = TransientTerm() == DiffusionTerm(coeff=D)
X, Y = mesh.faceCenters
phi.constrain(-1+d.faceValue, mesh.exteriorFaces) 
timeStepDuration = 10 * 0.9 * cellSize**2 / (2 * D)
steps = 10
from builtins import range
for step in range(steps):
    eq.solve(var=phi,
             dt=timeStepDuration) 
    if viewer is not None:
        viewer.plot() 
        
        
# %% FiPy trial
hydro.zeta = hydro.create_uniform_fipy_var(uniform_value=0.1, var_name='zeta')
hydro.theta = hydro.create_theta_from_zeta(hydro.zeta)
ZETA_DIRI_BC = -0.2
theta_face_values = hydro.ph_params.s1/hydro.ph_params.s2*(numerix.exp(hydro.ph_params.s2*ZETA_DIRI_BC) - numerix.exp(-hydro.ph_params.s2*hydro.dem.faceValue)) 
# constrain theta with those values
hydro.theta.constrain(theta_face_values, where=hydro.mesh.exteriorFaces)

# diffussivity mask: diff=0 in faces between canal cells. 1 otherwise.
# mask = 1*(hydro.canal_mask.arithmeticFaceValue > 1e-7)
s1 = hydro.ph_params.s1
s2 = hydro.ph_params.s2
t1 = hydro.ph_params.t1
t2 = hydro.ph_params.t2
# compute diffusivity. Cell Variable
diff_coeff = (
                t1/s1**(t2/s2)*(s1*numerix.exp(-hydro.dem*s2) + s2*hydro.theta)**(t2/s2-1) -
                t1*numerix.exp(-hydro.b*t2)/(s1*numerix.exp(-hydro.dem*s2) + s2*hydro.theta)
        )
# Apply mask
# diff_coeff.faceValue is a FaceVariable
diff_coeff_face = diff_coeff.faceValue

largeValue = 1e60
eq = (fp.TransientTerm() ==
        fp.DiffusionTerm(coeff=diff_coeff_face) + hydro.sourcesink
        # - fp.ImplicitSourceTerm(self.catch_mask_not * largeValue) +
        #self.catch_mask_not * largeValue * 0.
        )

NDAYS = 20
for i in range(NDAYS):
    residue = 1e10
    sweep_number = 0
    while residue > hydro.ph_params.fipy_desired_residual:
        residue = eq.sweep(var= hydro.theta, dt=hydro.ph_params.dt)
        sweep_number += 1
        
        if hydro.verbose:
            print(sweep_number, residue)

        if sweep_number > hydro.ph_params.max_sweeps:
            raise RuntimeError(
                'WTD computation did not converge in specified amount of sweeps')

    hydro.theta.updateOld()
    
zeta_after = numerix.log(s2*hydro.theta/s1 + numerix.exp(-s2*hydro.dem))/s2
# %%
