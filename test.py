#%%
import fipy as fp
import pandas as pd
from pathlib import Path
import rasterio
import numpy as np
from  fipy import numerix

#%% Prepare data

def read_raster(raster_filename):
    with rasterio.open(raster_filename) as raster:
        rst = raster.read(1)
    return rst

def sample_raster_from_coords(raster_filename:str, coords:np.ndarray) ->np.ndarray:
    raster_src = rasterio.open(raster_filename)
    return np.array([raster_value[0] for raster_value in raster_src.sample(coords)])

# def read_and_preprocess_dem(fn_dem):
#     dem = read_raster(fn_dem)
#     OUT_OF_BOUNDS_DEM_VALUE = -9999 # Negative value for catchment mask, etc. Later changed to positive.
#     dem[dem < -10] = OUT_OF_BOUNDS_DEM_VALUE
#     dem[np.where(np.isnan(dem))] = OUT_OF_BOUNDS_DEM_VALUE
#     dem[dem > 1e20] = OUT_OF_BOUNDS_DEM_VALUE  # just in case
#     return dem

parent_directory = Path(".")
fn_pointers = parent_directory.joinpath(r'file_pointers.xlsx')
filenames_df = pd.read_excel(
    fn_pointers, header=2, dtype=str, engine='openpyxl')
fn_dem = Path(filenames_df[filenames_df.Content == 'DEM'].Path.values[0])
mesh_fn = Path(filenames_df[filenames_df.Content == 'mesh'].Path.values[0])
fn_depth = Path(filenames_df[filenames_df.Content == 'depth'].Path.values[0])

SCALED = False

#%% Set up hydro

mesh = fp.Gmsh2D(mesh_fn)
mesh_centroids_coords = np.column_stack(mesh.cellCenters.value)

dem_value = sample_raster_from_coords(fn_dem, coords=mesh_centroids_coords)
dem = fp.CellVariable(name='dem', mesh=mesh, value=dem_value)

depth_value = sample_raster_from_coords(fn_depth, coords=mesh_centroids_coords)
depth_value[depth_value < 4] = 4  # cutoff depth
depth = fp.CellVariable(name='depth', mesh=mesh, value=depth_value, hasOld=False)

zeta_ini = -0.2
h = fp.CellVariable(name='h', mesh=mesh, hasOld=True, value=dem.value + zeta_ini)

s1 = 0.6
s2 = 0.5
t1 = 500
t2 = 2.5

# Scales
hc = 17.48
Sc = 1
qc = 0.003
xc = 1
tc = Sc*hc/qc
Tc = Sc*xc**2/tc

storage = s1 * numerix.exp(s2*(h - dem))
sourcesink = -0.0001
transmissivity = t1*( numerix.exp(t2*(h - dem)) - numerix.exp(-t2 * depth))

if SCALED:
    transmissivity = transmissivity/Tc
    storage = storage/Sc
    sourcesink = sourcesink/qc
    h = fp.CellVariable(name='h', mesh=mesh, hasOld=True, value=h.value/hc)


def set_eq_explicit(h, transmissivity, storage, sourcesink ):
    return (fp.TransientTerm() ==
              (transmissivity.faceValue * h.faceGrad).divergence / storage +
              sourcesink / storage
                      )

def set_eq_fixed_sto(h, transmissivity, storage, sourcesink ):
    sto = fp.CellVariable(mesh=mesh, value=storage.value)
    return (fp.TransientTerm(sto) ==
            fp.DiffusionTerm(coeff=transmissivity) +
              sourcesink
                      )

#%% Run hydro
max_sweeps = 10
niter = 24
dt = 10/niter
fipy_desired_residual = 1e-5

FIXED_STO = True

if SCALED:
    dt = dt/tc

for hour in range(niter):
    if FIXED_STO:
        eq = set_eq_fixed_sto(h, transmissivity, storage, sourcesink)
    else:
        eq = set_eq_explicit(h, transmissivity, storage, sourcesink)

    for sweep_number in range(max_sweeps):
        residue = eq.sweep(var=h, dt=dt)
        print(sweep_number, residue)

        if residue < fipy_desired_residual:
            break

    h.updateOld()
    print(f'---------- hour {hour} computed \n')
    print(f'mean storage: {storage.value.mean()} \n')

#%% Plot
if SCALED:
    h = h * hc
zeta = fp.CellVariable(name='zeta', mesh=mesh, value=h-dem, hasOld=True)
fp.Viewer(zeta)


