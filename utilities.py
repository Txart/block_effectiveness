# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 14:22:00 2018

@author: L1817
"""

import numpy as np
import copy
import random
import scipy.signal
import rasterio
import geopandas as gpd
from rasterio import features, fill
from scipy.spatial import cKDTree
import pandas as pd
import itertools
from operator import itemgetter

import preprocess_data

# %%


def get_already_built_block_positions(blocks_arr, labelled_canals):
    labelled_blocks = blocks_arr * labelled_canals
    labelled_blocks = labelled_blocks.astype(dtype=int)
    return tuple(labelled_blocks[labelled_blocks.nonzero()].tolist())


def get_sensor_loc_array_indices(sensor_loc_arr):
    locs = np.array(sensor_loc_arr.nonzero())
    loc_pairs = locs.transpose()
    return loc_pairs.tolist()


def peel_raster(raster, catchment_mask):
    """
    Given a raster and a mask, gets the "peeling" or "shell" of the raster. (Peeling here are points within the raster)
    Input:
        - raster: 2dimensional nparray. Raster to be peeled. The peeling is part of the raster.
        - catchment_mask: 2dim nparray of same size as raster. This is the fruit in the peeling.
    Output:
        - peeling_mask: boolean nparray. Tells where the peeling is.

    """
    # catchment mask boundaries by convolution
    conv_double = np.array([[0, 1, 1, 1, 0],
                            [1, 1, 1, 1, 1],
                            [1, 1, 0, 1, 1],
                            [1, 1, 1, 1, 1],
                            [0, 1, 1, 1, 0]])
    bound_double = scipy.signal.convolve2d(
        catchment_mask, conv_double, boundary='fill', fillvalue=False)
    peeling_mask = np.ones(shape=catchment_mask.shape, dtype=bool)
    peeling_mask[bound_double[2:-2, 2:-2] == 0] = False
    peeling_mask[bound_double[2:-2, 2:-2] == 20] = False

    peeling_mask = (catchment_mask*peeling_mask) > 0

    return peeling_mask


# NEW 23.11. ONE STEP OR MOVEMENT IN THE SIMULATED ANNEALING.
def switch_one_dam(oWTcanals, surface_canals, currentWTcanals, block_height, dams_location, n_canals, CNM):
    """
        Randomly chooses which damn to take out and where to put it again.
    Computes which are "prohibited nodes", where the original water level of the canal is
    lower than the current (i.e., has been affected by the placement of a dam), before making the
    otherwise random decision of locating the dam.

    OUTPUT
        - new_wt_canal: wl in canal after removing old and placing a new dam.
    """

    # Select dam to add
    # Check for prohibited nodes. Minor problem: those points affected by the dam which will be soon removed are also prohibited
    prohibited_node_list = [i for i, _ in enumerate(
        oWTcanals) if oWTcanals[i] < currentWTcanals[i]]
    candidate_node_list = [e for e in range(
        0, n_canals) if e not in prohibited_node_list]
    random.shuffle(candidate_node_list)  # Happens in-place.
    dam_to_add = candidate_node_list[0]

    # Select dam to remove
    # Shuffle to select which canal to remove. Happens in place.
    random.shuffle(dams_location)
    dam_to_remove = dams_location[0]

    dams_location.remove(dam_to_remove)
    dams_location.append(dam_to_add)

    # Compute new wt in canals with this configuration of dams
    new_wt_canal = place_dams(
        oWTcanals, surface_canals, block_height, dams_location, CNM)

    return new_wt_canal


def PeatV_weight_calc(canal_mask):
    """ Computes weights (associated to canal mask) needed for peat volume compt.

    input: canal_mask -- np.array of dim (nx,ny). 0s where canals or outside catchment, 1s in the rest.


    output: np.array of dim (nx,ny) with weights to compute energy of sim anneal
    """

    xdim = canal_mask.shape[0]
    ydim = canal_mask.shape[1]

    # Auxiliary array of ones and zeros.
    # Extra rows and cols of zeros to deal with boundary cases
    arr_of_ones = np.zeros((xdim+2, ydim+2))
    arr_of_ones[1:-1, 1:-1] = canal_mask  # mask

    # Weights array
    weights = np.ones((xdim, ydim))

    # Returns number of non-zero 0th order nearest neighbours
    def nn_squares_sum(arr, row, i):
        nsquares = 0
        if ((arr[row, i] + arr[row-1, i] + arr[row-1, i-1] + arr[row, i-1]) == 4):
            nsquares += 1
        if ((arr[row, i] + arr[row, i-1] + arr[row+1, i-1] + arr[row+1, i]) == 4):
            nsquares += 1
        if ((arr[row, i] + arr[row+1, i] + arr[row+1, i+1] + arr[row, i+1]) == 4):
            nsquares += 1
        if ((arr[row, i] + arr[row, i+1] + arr[row-1, i+1] + arr[row-1, i]) == 4):
            nsquares += 1
        return nsquares

    for j, row in enumerate(arr_of_ones[1:-1, 1:-1]):
        for i, _ in enumerate(row):
            weights[j, i] = nn_squares_sum(arr_of_ones, j+1, i+1)

    return weights


def PeatVolume(weights, Z):
    """Computation of dry peat volume. Energy for the simulated annealing.
    INPUT:
        - weights: weights as computed from nn_squares_sum function
        - Z: array of with values = surface dem elevation - wt
    OUTPUT:
        - Dry peat volume. Units: ha x m. The ha thing is coming from the pixel size being 100x100m and dx=dy=1.
        On the other hand, water table depth is in m.
    """

    # This is good code for the future.
#    sur = np.multiply(surface, weights) # element-wise multiplication
#    gwt = np.multiply(gwt, weights) # element-wise multiplication
#    gehi_sur = np.sum(sur) # element-wise sum
#    gehi_gwt = np.sum(gwt)
    # all operations linear, so same as doing them with Z=surface-gwt
    zet = np.multiply(Z, weights)
    z_sum = np.sum(zet)

    # Carefull with gwt over surface!
#    dry_peat_volume = .25 * (ygrid[1]-ygrid[0])*(xgrid[1]-xgrid[0]) * (gehi_sur - gehi_gwt)

    dry_peat_volume = .25 * z_sum

    return dry_peat_volume


def print_time_in_mins(time):
    if time > 60 and time < 3600:
        print("Time spent: ", time/60.0, "minutes")
    elif time > 3600:
        print("Time spent: ", time/60.0, "hours")
    else:
        print("Time spent: ", time, "seconds")


def place_dams(originalWT, srfc, block_height, dams_to_add, CNM):
    """ Takes original water level in canals and list of nodes where to put blocks. Returns updated water level in canals.

    Input:
        - originalWT: list. Original water level in canals.
        - srfc: list. DEM value at canals.
        - block_height: float. Determines the new value of the water level as: new_value = surface[add_can] - block_height.
        - dams_to_add: list of ints. positions of dam to add.
        - CNM: propagation or canal adjacency (sparse) matrix.

    Output:
        - wt: list. Updated water level in canals.
    """

    def addDam(wt, surface, block_height, add_dam, CNM):
        """ Gets a single canal label and returns the updated wt corresponding to building a dam in that canal
        """
        add_height = surface[add_dam] - block_height
        list_of_canals_to_add = [add_dam]

        while len(list_of_canals_to_add) > 0:
            # try to avoid numpyfication
            list_of_canals_to_add = list(list_of_canals_to_add)
            add_can = list_of_canals_to_add[0]

            if wt[add_can] < add_height:  # condition for update
                wt[add_can] = add_height
                # look for propagation in matrix
                canals_prop_to = CNM[add_can].nonzero()[1].tolist()
                # append canals to propagate to. If none, it appends nothing.
                list_of_canals_to_add = list_of_canals_to_add + canals_prop_to

            # remove add_can from list
            list_of_canals_to_add = list_of_canals_to_add[1:]

        return wt

    wt = copy.deepcopy(originalWT)  # Always start with original wt.

#    if type(dams_to_add) != list:
#        print("dams_to_add needs to be list type")

    for add_can in dams_to_add:
        wt = addDam(wt, srfc, block_height, add_can, CNM)

    return wt


def from_raster_pos_to_LatLong(positions_in_canal_network, c_to_r_list, can_network_raster_fn):

    import rasterio
    rows = [c_to_r_list[pos][0] for pos in positions_in_canal_network]
    cols = [c_to_r_list[pos][1] for pos in positions_in_canal_network]
    # read metadata from input raster to copy it and have the same projection etc
    src = rasterio.open(can_network_raster_fn)

    lats, longs = rasterio.transform.xy(
        transform=src.profile['transform'], rows=rows, cols=cols)

    latlongs = [(lats[i], longs[i]) for i in range(0, len(lats))]

    return latlongs


def map_lc_number_to_lc_coef_in_co2_emission_formula(lc_raster):
    # Coefficients taken from file 21092020_Formula for WTD conversion.xlsx
    # created by Imam and uploaded to Asana

    lc_multiplicative_coef_co2 = np.zeros(shape=lc_raster.shape)
    lc_additive_coef_co2 = np.zeros(shape=lc_raster.shape)

    lc_multiplicative_coef_co2[(lc_raster == 2002) | (
        lc_raster == 20041) | (lc_raster == 20051)] = -98  # Degraded forests
    lc_multiplicative_coef_co2[lc_raster == 2005] = -98  # Primary forests
    lc_multiplicative_coef_co2[lc_raster == 2006] = -69  # Acacia plantation
    lc_multiplicative_coef_co2[(lc_raster == 2007) | (lc_raster == 20071) | (lc_raster == 20091) |
                               (lc_raster == 20092) | (lc_raster == 20093) | (lc_raster == 2012) |
                               (lc_raster == 20121) | (lc_raster == 2014) | (lc_raster == 20141) |
                               (lc_raster == 50011)] = -84  # Deforested peatlands
    lc_multiplicative_coef_co2[lc_raster == 2010] = -77.07  # Other Plantations

    lc_additive_coef_co2[(lc_raster == 2002) | (lc_raster == 20041) | (
        lc_raster == 20051)] = 0  # Degraded forests
    lc_additive_coef_co2[lc_raster == 2005] = 0  # Primary forests
    lc_additive_coef_co2[lc_raster == 2006] = 21  # Acacia plantation
    lc_additive_coef_co2[(lc_raster == 2007) | (lc_raster == 20071) | (lc_raster == 20091) |
                         (lc_raster == 20092) | (lc_raster == 20093) | (lc_raster == 2012) |
                         (lc_raster == 20121) | (lc_raster == 2014) | (lc_raster == 20141) |
                         (lc_raster == 50011)] = 9  # Deforested peatlands
    lc_additive_coef_co2[lc_raster == 2010] = 19.8  # Other Plantations

    return lc_multiplicative_coef_co2, lc_additive_coef_co2


def map_lc_number_to_lc_coef_in_subsidence_formula(lc_raster):
    # Coefficients taken from file 21092020_Formula for WTD conversion.xlsx
    # created by Imam and uploaded to Asana

    lc_multiplicative_coef_subsi = np.zeros(shape=lc_raster.shape)
    lc_additive_coef_subsi = np.zeros(shape=lc_raster.shape)

    lc_multiplicative_coef_subsi = np.zeros(shape=lc_raster.shape)
    lc_multiplicative_coef_subsi[(lc_raster == 2002) | (
        lc_raster == 20041) | (lc_raster == 20051)] = -6.54  # Degraded forests
    lc_multiplicative_coef_subsi[lc_raster == 2005] = -7.06  # Primary forests
    lc_multiplicative_coef_subsi[lc_raster ==
                                 2006] = -4.98  # Acacia plantation
    lc_multiplicative_coef_subsi[(lc_raster == 2007) | (lc_raster == 20071) | (lc_raster == 20091) |
                                 (lc_raster == 20092) | (lc_raster == 20093) | (lc_raster == 2012) |
                                 (lc_raster == 20121) | (lc_raster == 2014) | (lc_raster == 20141) |
                                 (lc_raster == 50011)] = -5.98  # Deforested peatlands
    # Other Plantations, same as Acacia
    lc_multiplicative_coef_subsi[lc_raster == 2010] = -4.98

    lc_additive_coef_subsi = np.zeros(shape=lc_raster.shape)
    lc_additive_coef_subsi[(lc_raster == 2002) | (lc_raster == 20041) | (
        lc_raster == 20051)] = 0.35  # Degraded forests
    lc_additive_coef_subsi[lc_raster == 2005] = 0  # Primary forests
    lc_additive_coef_subsi[lc_raster == 2006] = 1.5  # Acacia plantation
    lc_additive_coef_subsi[(lc_raster == 2007) | (lc_raster == 20071) | (lc_raster == 20091) |
                           (lc_raster == 20092) | (lc_raster == 20093) | (lc_raster == 2012) |
                           (lc_raster == 20121) | (lc_raster == 2014) | (lc_raster == 20141) |
                           (lc_raster == 50011)] = 0.69  # Deforested peatlands
    # Other Plantations, same as Acacia
    lc_additive_coef_subsi[lc_raster == 2010] = 1.5

    return lc_multiplicative_coef_subsi, lc_additive_coef_subsi


def compute_co2_from_WTD(wtd, lc_multiplicative_coef_co2, lc_additive_coef_co2):
    wtd_hooijer_capped = wtd[:]
    wtd_carlson_capped = wtd[:]

    # WTD are capped because wtd for the co2 model in papers do not range further than -1.25
    # papers: hooijer, carlson, evans. From Imam's formula as discussed in Asana
    wtd_hooijer_capped[wtd < -1.25] = -1.25
    wtd_carlson_capped[wtd < -1.1] = -1.1

    co2_hooijer = lc_additive_coef_co2 + \
        lc_multiplicative_coef_co2 * wtd_hooijer_capped  # tCO2eq/ha/yr
    co2_carlson = lc_additive_coef_co2 + \
        lc_multiplicative_coef_co2 * wtd_carlson_capped

    co2 = co2_hooijer
    carlson_mask = np.where(lc_multiplicative_coef_co2 == -77.07)
    co2[carlson_mask] = co2_carlson[carlson_mask]
    co2 = co2/365  # tCO2eq/ha/yr -> tCO2eq/ha/day

    return co2


def compute_subsi_from_WTD(wtd, lc_multiplicative_coef_subsi, lc_additive_coef_subsi):
    wtd_hooijer_capped = wtd[:]
    wtd_evans_capped = wtd[:]

    # WTD are capped because wtd for the co2 model in papers do not range further than -1.25
    # papers: hooijer, carlson, evans. From Imam's formula as discussed in Asana
    wtd_hooijer_capped[wtd < -1.25] = -1.25
    wtd_evans_capped[wtd < -1.2] = -1.2

    subsi_hooijer = lc_additive_coef_subsi + \
        lc_multiplicative_coef_subsi * wtd_hooijer_capped
    subsi_evans = lc_additive_coef_subsi + \
        lc_multiplicative_coef_subsi * wtd_evans_capped

    subsi = subsi_hooijer
    evans_mask = np.where(lc_multiplicative_coef_subsi == -6.54)
    subsi[evans_mask] = subsi_evans[evans_mask]

    subsi = lc_additive_coef_subsi + lc_multiplicative_coef_subsi * wtd  # cm/yr
    subsi = subsi/100/365  # cm/yr -> m/day

    return subsi


def write_raster_multiband(nbands, rasters_array, STUDY_AREA, out_filename, ref_filename=r"data/Strat4/DTM_metres_clip.tif"):
    # Rasters_array should be a numpy array with the form [bands, rows, col]
    # This means that if there are 4 bands in the raster, the shape of the array
    # should be (4, nrows, ncols)
    # Metadata is copied from the raster at ref_filename

    # src file is needed to output with the same metadata and attributes
    with rasterio.open(ref_filename) as src:
        profile = src.profile
    _, _, ref_raster, _, _, _, _ = preprocess_data.read_preprocess_rasters(
        STUDY_AREA, ref_filename, ref_filename, ref_filename, ref_filename, ref_filename, ref_filename, ref_filename)

    if nbands != rasters_array.shape[0]:
        raise ValueError("Number of bands does not match  raster array shape")

    profile.update(nodata=None)  # overrun nodata value given by input raster
    # Shape of rater is changed for hydro simulation inside read_preprocess_rasters. Here we take that shape to output consistently.
    profile.update(width=ref_raster.shape[1])
    profile.update(height=ref_raster.shape[0])
    profile.update(count=nbands)
    # instead of 64. To save space, we don't need so much precision. float16 is not supported by GDAL, check: https://github.com/mapbox/rasterio/blob/master/rasterio/dtypes.py
    profile.update(dtype='float32')

    with rasterio.open(out_filename, 'w', **profile) as dst:
        dst.write(rasters_array.astype(dtype='float32'))

    return 0


def write_raster_from_array(array_to_rasterize, out_filename, template_raster_filename):
    with rasterio.open(template_raster_filename) as src:
        profile = src.profile
    
    profile.update(nodata=None)  # overrun nodata value given by input raster
    # instead of 64. To save space, we don't need so much precision. float16 is not supported by GDAL, check: https://github.com/mapbox/rasterio/blob/master/rasterio/dtypes.py
    profile.update(dtype='float32')
    
    with rasterio.open(out_filename, 'w', **profile) as dst:
        dst.write_band(1, array_to_rasterize.astype(dtype='float32'))

def invert_dictionary(dictionary, are_values_unique=True):
    """Invert an input dictionary, with the option to keep all non-unique values

    Args:
        dictionary (dict): Dictionary to invert.
        are_values_unique (bool, optional): If True, then the inverted dictionary will only have one value per key. If False, it creates a list of all the values. Defaults to True.

    Returns:
        dict: Inverted dictionary.
    """
    if are_values_unique:
        inv_dict = {v: k for k, v in dictionary.items()}

    else:  # more than one value per key; keep all values
        inv_dict = {}
        for k, v in dictionary.items():
            inv_dict[v] = inv_dict.get(v, []) + [k]

    return inv_dict


def is_value_of_variable_possible(variable, possible_values):
    if variable not in possible_values:
        raise ValueError(
            'The value of a variable is not among its possible values. Aborting')
        

def find_nearest_neighbour_in_array_of_points(array_of_points):
    # finds nearest neighbour of each element in an array of points
    # array_of_points must have the following shape:
    # np.array([[x_0, y_0],
    #           [x_1, y_1],
    #           ...        ])
    # returns (distances to nn, indices of nn) for each element in array
    tree = cKDTree(array_of_points)
    distances, indices = tree.query(array_of_points, k=2) # k=2 means we get up to the second neighbor
    
    # the closest points are themselves (k=1)
    # So we need to chose the first neighbour (k=2)
    return distances[:,1], indices[:,1]

def find_nearest_point_in_other_array(source_point_array, target_point_array):
    # For each point source point, finds the closest target point
    # source and target should be np.arrays with the following shape:
    # np.array([[x_0, y_0],
    #           [x_1, y_1],
    #           ...        ])
    # returns (distance to target point, index of target point)
    # for each elemen in the source point array
    btree = cKDTree(target_point_array)
    dist, idx = btree.query(source_point_array, k=1)
    return dist, idx
    
    

def find_nearest_point_in_other_geodataframe(gdA, gdB):
    # From https://gis.stackexchange.com/questions/222315/geopandas-find-nearest-point-in-other-dataframe
    # It looks for points in gdB that are closest to points of gdA
    # So, the output will have the dimension of gdA. 
    # Both geodataframes need to have a column called 'geometry' that contains only POINTS (not MULTIPOINTS)
    nA = np.array(list(gdA.geometry.apply(lambda x: (x.x, x.y))))
    nB = np.array(list(gdB.geometry.apply(lambda x: (x.x, x.y))))
    dist, idx = find_nearest_point_in_other_array(nA, nB)
    gdB_nearest = gdB.iloc[idx].index
    # output df
    df = pd.DataFrame(index=gdA.index)
    df['A'] = np.array(gdA.index)
    df['B'] = np.array(gdB_nearest)
    df['dist'] = np.array(dist)

    return df

def multilinestring_to_linestring(geodataframe):
    output_gdf = copy.deepcopy(geodataframe)
    for index, row in geodataframe.iterrows():
        output_gdf.loc[index, "geometry"] = row.geometry[0]
        
    return output_gdf

def find_distance_of_points_in_geodataframe_to_nearest_line_in_geodataframe(gdf_points, gdf_lines):
    # Adapted from: https://gis.stackexchange.com/questions/222315/finding-nearest-point-in-other-geodataframe-using-geopandas
    A = np.concatenate(
        [np.array(geom.coords) for geom in gdf_points.geometry.to_list()])
    B = [np.array(geom.coords) for geom in gdf_lines.geometry.to_list()]
    B_ix = tuple(itertools.chain.from_iterable(
        [itertools.repeat(i, x) for i, x in enumerate(list(map(len, B)))]))
    B = np.concatenate(B)
    ckd_tree = cKDTree(B)
    dist, idx = ckd_tree.query(A, k=1)
    idx = itemgetter(*idx)(B_ix)

    # output df
    df = pd.DataFrame(index=gdf_points.index)
    df['point'] = np.array(gdf_points.index)
    df['line'] = np.array(gdf_lines.loc[list(idx)].index)
    df['distance'] = dist
    
    return df

def extract_coords_from_geopandas_dataframe(gpd_dataframe:gpd.GeoDataFrame) -> np.ndarray:
    # gpd_dataframe needs to have a column of simple geometries
    # If MULTI geometries present, use something like gpd.explode()
    x_coords = gpd_dataframe.geometry.x.to_numpy()
    y_coords = gpd_dataframe.geometry.y.to_numpy()
    return np.column_stack((x_coords, y_coords))

def convert_networkx_graph_to_dataframe(graph): 
 	dict_nodes_to_attributes = dict(graph.nodes.data(True))
 	df = pd.DataFrame.from_dict(dict_nodes_to_attributes, orient='index')

 	return df

def sample_raster_from_coords(raster_filename:str, coords:np.ndarray) ->np.ndarray:
    raster_src = rasterio.open(raster_filename)
    return np.array([raster_value[0] for raster_value in raster_src.sample(coords)])

def _rasterize_geodataframe_nearest_neighbour_interpolation(gdf, gdf_column_name, template_raster_fn, out_raster_fn):
    # Open template raster and copy its metadata
    trst = rasterio.open(template_raster_fn)
    template_metadata = trst.meta.copy()
    
    # Extract values and their position from geodataframe 
    shapes = ((geom, val) for geom, val in zip(gdf['geometry'], gdf[gdf_column_name]))

    # Open and write raster
    with rasterio.open(out_raster_fn, 'w+', **template_metadata) as out:
        out_arr = out.read(1)
        raster = features.rasterize(shapes=shapes, fill=0, out=out_arr, transform=out.transform)

        return raster
    
    
def _rasterize_geodataframe_linear_interpolation(gdf, gdf_column_name, template_raster_fn, out_raster_fn):
    # Inspiration from 
    # https://stackoverflow.com/questions/71665265/creating-a-lat-lon-grid-from-lat-lon-point-data-of-a-given-variable-in-python
    from scipy import interpolate
    from shapely.geometry import Point
    
    # Open template raster and copy its metadata
    trst = rasterio.open(template_raster_fn)
    template_raster = trst.read(1)
    template_metadata = trst.meta.copy()

    xs, ys = gdf.geometry.x, gdf.geometry.y
    coords = np.column_stack((np.array(xs), np.array(ys)))
    values = gdf[gdf_column_name]

    # define raster resolution based on template raster
    rRes = template_metadata['transform'][0]

    # create coord ranges over the desired raster extension
    xRange = np.arange(xs.min(), xs.max(), rRes)
    yRange = np.arange(ys.min(), ys.max(), rRes)

    # create arrays of x,y over the raster extension
    gridX, gridY = np.meshgrid(xRange, yRange)

    # interpolate over the grid
    gridPh = interpolate.griddata(coords, values, (gridX, gridY), method='linear')
    gridPh = gridPh[::-1] # For some reason, it's upside down

    area_mask = 1*(template_raster > -1)
    # Convert 0s in area_mask to NaNs
    area_mask = area_mask.astype('float') # needs to be float because Nan is float
    area_mask[area_mask == 0] = np.nan

    return gridPh * area_mask

def rasterize_geodataframe_with_template(template_raster_fn:str, out_raster_fn:str,
                                         gdf:gpd.GeoDataFrame, gdf_column_name:'str',
                                         interpolation='nearest_neighbour')->np.ndarray:
    """Takes geodataframe and template raster and returns raster of the geodataframe values.

    Args:
        template_raster_fn (str): filename of the template raster from which metadata is copied
        out_raster_fn (str): filename outpur raster
        gdf (gpd.GeoDataFrame): geodataframe with values to rasterize. Must contain a geometry column.
        gdf_column_name (str): Name of the geodataframe column to burn in the raster
        interpolation (str): Options are "nearest_neighbour" and "linear". nn takes the closest
            value in the gdf and burns it directly. Linear does a linear interpolation

    Returns:
        raster
    """    
    
    if interpolation == 'nearest_neighbour':
        return _rasterize_geodataframe_nearest_neighbour_interpolation(gdf, gdf_column_name,
                                                         template_raster_fn, out_raster_fn)
    elif interpolation == 'linear':
        return _rasterize_geodataframe_linear_interpolation(gdf, gdf_column_name, template_raster_fn, out_raster_fn)
    
    else:
        raise ValueError('interpolation type not understood')

