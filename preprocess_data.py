# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 09:57:45 2018

@author: L1817
"""

import numpy as np
import rasterio
import scipy.sparse
import pandas as pd

#%%

def read_params(fn=r"/home/inaki/GitHub/dd_winrock/data/params.xlsx"):
    df = pd.read_excel(fn)
    return df


def peat_depth_map(peat_depth_type_arr):
    peat_depth_arr = np.ones(shape=peat_depth_type_arr.shape)
    # information from excel
    peat_depth_arr[peat_depth_type_arr==1] = 2. # depth in meters.
    peat_depth_arr[peat_depth_type_arr==2] = 2.
    peat_depth_arr[peat_depth_type_arr==3] = 4.
    peat_depth_arr[peat_depth_type_arr==4] = 0.
    peat_depth_arr[peat_depth_type_arr==5] = 0.
    peat_depth_arr[peat_depth_type_arr==6] = 2.
    peat_depth_arr[peat_depth_type_arr==7] = 4.
    peat_depth_arr[peat_depth_type_arr==8] = 8.
    
    return peat_depth_arr

def read_raster(raster_filename):
    with rasterio.open(raster_filename) as raster:
        rst = raster.read(1)
    return rst

def preprocess_can_arr_raster(can_arr):
    can_arr[can_arr < 0.5] = 0
    can_arr[abs(can_arr) > 0.5] = 1
    can_arr = np.array(can_arr, dtype=int)
    return can_arr

#dupli
def preprocess_dem(dem):
    dem[dem <-10] = -9999.0
    dem[np.where(np.isnan(dem))] = -9999.0
    dem[dem > 1e20] = -9999.0 # just in case
    return dem

def preprocess_peat_type(peat_type_arr, dem):
    peat_type_arr[peat_type_arr < 0] = -1
    # fill some nodata values to get same size as dem
    peat_type_arr[(np.where(dem>0.1) and np.where(peat_type_arr <0.1))] = 1.
    return peat_type_arr
    
def preprocess_peat_depth(peat_depth_arr, dem):
    peat_depth_arr[peat_depth_arr < 0] = -1
    peat_depth_arr = peat_depth_map(peat_depth_arr) # translate number keys to depths
    # fill some nodata values to get same size as dem
    peat_depth_arr[(np.where(dem>0.1) and np.where(peat_depth_arr <0.1))] = 1.
    return peat_depth_arr

def mask_non_acquatic_blocks(blocks_arr, can_arr):
    return blocks_arr*can_arr
    
def resize_study_area(sa, raster):
    return raster[sa[0][0]:sa[0][1], sa[1][0]:sa[1][1]]    
      
def read_preprocess_rasters(sa, wtd_old_rst_fn, can_rst_fn, dem_rst_fn, peat_type_rst_fn, peat_depth_rst_fn, blocks_rst_fn, sensor_loc_fn):
    """
    Deals with issues specific to each  input raster.
    Corrects nodatas, resizes to selected study area, etc.
    sa: integers giving array slicing to constrain. Study area.
    """

    wtd_old = read_raster(wtd_old_rst_fn)
    can_arr = read_raster(can_rst_fn)
    dem = read_raster(dem_rst_fn)
    peat_type_arr = read_raster(peat_type_rst_fn)
    peat_depth_arr = read_raster(peat_depth_rst_fn)
    blocks_arr = read_raster(blocks_rst_fn)
    sensor_loc_arr = read_raster(sensor_loc_fn)
        
        
    can_arr = preprocess_can_arr_raster(can_arr)  #Get mask of canals: 1 where canals exist, 0 otherwise
    dem = preprocess_dem(dem)   # Convert from numpy no data to -9999.0
    peat_type_arr = preprocess_peat_type(peat_type_arr, dem) # control nodata values, impose same size as dem
    peat_depth_arr = preprocess_peat_depth(peat_depth_arr, dem) # nodatas, same size as dem and give peat depth map values
    blocks_arr = mask_non_acquatic_blocks(blocks_arr, can_arr) # only useful blocks are those that are in water! (I.e., that coincide with a water pixel)

    # Apply study area restriction
    wtd_old = resize_study_area(sa, wtd_old)
    dem = resize_study_area(sa, dem)
    can_arr = resize_study_area(sa, can_arr)
    peat_depth_arr = resize_study_area(sa, peat_depth_arr)
    peat_type_arr = resize_study_area(sa, peat_type_arr)
    blocks_arr = resize_study_area(sa, blocks_arr)
    sensor_loc_arr = resize_study_area(sa, sensor_loc_arr)

    
    return can_arr, wtd_old, dem, peat_type_arr, peat_depth_arr, blocks_arr, sensor_loc_arr

def read_preprocess_landcover(sa, lc_fn):
    lc = read_raster(lc_fn)
    lc[lc<0] = 0 #NoData Values
    lc = resize_study_area(sa, lc)
    return lc
    
    
# Build the adjacency matrix up
def _prop_to_neighbours(pixel_coords, rasterized_canals, dem, threshold=0.0):
    """Given a pixel where a canals exists, return list of canals that it would propagate to.
    Info taken for allowing propagation: DEM height.
    Threshold gives strictness of the propagation condition: if 0, then only strictly increasing water tables are propagated.
    """
    padded_can_arr = np.pad(rasterized_canals,pad_width=1,mode='constant')
    padded_dem = np.pad(dem,pad_width=1,mode='constant')
    pc0 = pixel_coords[0] + 1; pc1 = pixel_coords[1] + 1 # plus one bc of padding
    prop_to = []

    candidate_coords_list = ([(pc0-1,pc1-1), (pc0-1,pc1), (pc0-1,pc1+1),
                              (pc0,pc1-1),               (pc0,pc1+1),
                              (pc0+1,pc1-1), (pc0+1,pc1), (pc0+1,pc1+1)])
    for cc in candidate_coords_list:
        if padded_can_arr[cc] > 0: # pixel corresponds to a canal
            if padded_dem[cc] - padded_dem[pc0,pc1] > -threshold:
                prop_to.append(int(padded_can_arr[cc]))      
    return prop_to

def label_canal_pixels(can_arr, dem):
    # Convert labels of canals to 1,2,3...
    aux =can_arr.flatten()
    can_flat_arr = np.array(aux)
    aux_dem = dem.flatten()
    
    counter = 1
    for i, value in enumerate(aux):
        if value > 0 and aux_dem[i] > 0: # if there is a canal in that pixel and if we have a dem value for that pixel. (Missing dem data points are labelled as -9999)
            can_flat_arr[i] = counter
            counter += 1
    labelled_canals = can_flat_arr.reshape(can_arr.shape) # contains labels and positions of canals
    
    
    return labelled_canals
    


def gen_can_matrix_and_label_map(labelled_canals, dem):
    """ Gets canal RASTER FILE and generates adjacency matrix.
    
    Input:
        - labelled canals: output oof label_canal_pixels
        -dem: dem is used as a template
        
    Output:
        - matrix: canal adjacency matrix NumPy array.
        - out_arr: canal raster in NumPy array.
        - can_to_raster_list: list. Pixels corresponding to each canal.
    """

    n_canals = int(labelled_canals.max() + 1) 
    # compute matrix AND dict
    matrix = scipy.sparse.lil_matrix((n_canals, n_canals)) # lil matrix is good to build it incrementally

    c_to_r_list = [0] * n_canals
    for coords, label in np.ndenumerate(labelled_canals):
        if labelled_canals[coords] > 0: # if coords correspond to a canal.
            c_to_r_list[int(label)] = coords # label=0 is not a canal. But do not delete it, otherwise everything would be corrido.
            propagated_to = _prop_to_neighbours(coords, labelled_canals, dem, threshold=0.0) 
            for i in propagated_to:
                matrix[int(label), i] = 1 # adjacency matrix of the directed graph
    
    matrix_csr = matrix.tocsr() # compressed row format is more efficient for later
    matrix_csr.eliminate_zeros() # happens in place. Frees disk usage.
    
    return matrix_csr, c_to_r_list


def get_array_indices_by_rows(array_shape):
    rows, cols = np.indices(array_shape)
    indices = np.array(list(zip(rows.flatten(), cols.flatten())))
    final_shape = list(array_shape) + [2] # The 2 comes from the 2D
    
    return indices.reshape(final_shape)

def nearest_neighbors_mask_from_coordinates(array_shape, points_coordinates):
    """
    Takes the shape of an array and a set of points within the array and returns a mask
    that contains, for each index in the array, the label of the closest point
    (and closeness is measured by Euclidean distance).
    In this particular application, it is used to know what weather station to 
    pick the data from for each pixel.
    NOTE: When two positions are exactly at the same distance, the first
    points_coordinate in order is chosen

    Parameters
    ----------
    array_shape : tuple
        The shape of the mask. Computed with numpy's shape
    points_coordinates : list of coordinate tuples
        Each key is used as a value in the mask array. The tuples are used
        to compute the distaance to other elements of the array

    Returns
    -------
    mask : np.array
    
    mask_dictionary: dict
        Keys are the values in the returned mask, and values are the corresponding
        point coordinates from the input

    """
    
    from scipy.spatial import distance
    
    # Check that point coordinates lie inside the array
    mask = np.zeros(shape=array_shape)
    for coord in points_coordinates:
        try:
            mask[coord[0], coord[1]]
        except:
            raise ValueError("the coordinates are out of the array's bounds")
        
    indices_by_row = get_array_indices_by_rows(array_shape)
    
    # Create output dictionary of points
    mask_dictionary = {}
    for npoint, point in enumerate(points_coordinates):
        mask_dictionary[npoint] = point
    
    for row_n, row in enumerate(indices_by_row):
        dists = distance.cdist(row, np.array(points_coordinates))
        mask[row_n] = np.argmin(dists.T, axis=0)
    
    return mask, mask_dictionary



   # Activate this to run this module as main         
#if __name__== '__main__':
#    datafolder = r"C:\Users\L1817\Dropbox\PhD\Computation\Indonesia_WaterTable\Winrock\Canal_Block_Data\GIS_files\Stratification_layers"
#    preprocess_datafolder = r"C:\Users\L1817\Dropbox\PhD\Computation\Indonesia_WaterTable\Winrock\preprocess"
#    dem_rst_fn = r"\dem_clip_cean.tif"
#    can_rst_fn = r"\can_rst_clipped.tif"
#    
#    cm, cr, c_to_r_list = gen_can_matrix_and_raster_from_raster(can_rst_fn=preprocess_datafolder+can_rst_fn,
#                                                                dem_rst_fn=datafolder+dem_rst_fn)
#    # Pickle outcome!
##    import pickle
#    pickle_folder = r"C:\Users\L1817\Winrock"
#    with open(pickle_folder + r'\50x50_DEM_and_canals.pkl', 'w') as f:
#        pickle.dump([cm, cr, c_to_r_list], f)

