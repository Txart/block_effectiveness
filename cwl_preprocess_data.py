# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 09:57:45 2018

@author: L1817
"""
#%%
import numpy as np
import rasterio
import scipy.sparse
import pandas as pd
import geopandas
import networkx as nx
import pickle
from shapely.geometry import LineString, Point

import cwl_utilities

# %%


def read_params(fn=r"/home/inaki/GitHub/dd_winrock/data/params.xlsx"):
    df = pd.read_excel(fn)
    return df


def peat_depth_map(peat_depth_type_arr):
    peat_depth_arr = np.ones(shape=peat_depth_type_arr.shape)
    # information from excel
    peat_depth_arr[peat_depth_type_arr == 1] = 2.  # depth in meters.
    peat_depth_arr[peat_depth_type_arr == 2] = 2.
    peat_depth_arr[peat_depth_type_arr == 3] = 4.
    peat_depth_arr[peat_depth_type_arr == 4] = 0.
    peat_depth_arr[peat_depth_type_arr == 5] = 0.
    peat_depth_arr[peat_depth_type_arr == 6] = 2.
    peat_depth_arr[peat_depth_type_arr == 7] = 4.
    peat_depth_arr[peat_depth_type_arr == 8] = 8.

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


def preprocess_dem(dem):
    dem[dem < -10] = -9999.0
    dem[np.where(np.isnan(dem))] = -9999.0
    dem[dem > 1e20] = -9999.0  # just in case
    return dem


def preprocess_peat_type(peat_type_arr, dem):
    peat_type_arr[peat_type_arr < 0] = -1
    # fill some nodata values to get same size as dem
    peat_type_arr[(np.where(dem > 0.1) and np.where(peat_type_arr < 0.1))] = 1.
    return peat_type_arr


def preprocess_peat_depth(peat_depth_arr, dem):
    peat_depth_arr[peat_depth_arr < 0] = -1
    # translate number keys to depths
    peat_depth_arr = peat_depth_map(peat_depth_arr)
    # fill some nodata values to get same size as dem
    peat_depth_arr[(np.where(dem > 0.1) and np.where(
        peat_depth_arr < 0.1))] = 1.
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

    # Get mask of canals: 1 where canals exist, 0 otherwise
    can_arr = preprocess_can_arr_raster(can_arr)
    dem = preprocess_dem(dem)   # Convert from numpy no data to -9999.0
    # control nodata values, impose same size as dem
    peat_type_arr = preprocess_peat_type(peat_type_arr, dem)
    # nodatas, same size as dem and give peat depth map values
    peat_depth_arr = preprocess_peat_depth(peat_depth_arr, dem)
    # only useful blocks are those that are in water! (I.e., that coincide with a water pixel)
    blocks_arr = mask_non_acquatic_blocks(blocks_arr, can_arr)

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
    lc[lc < 0] = 0  # NoData Values
    lc = resize_study_area(sa, lc)
    return lc


# Build the adjacency matrix up
def _prop_to_neighbours(pixel_coords, rasterized_canals, dem, threshold=0.0, is_diagonal_a_neighbor=False):
    """Given a pixel where a canals exists, return list of canals that it would propagate to.
    Info taken for allowing propagation: DEM height.
    Threshold gives strictness of the propagation condition: if 0, then only strictly increasing water tables are propagated.
    If is_diagonal_a_neighbor=True, diagonally adjacent pixels are seen as neighbours.
    """
    padded_can_arr = np.pad(rasterized_canals, pad_width=1, mode='constant')
    padded_dem = np.pad(dem, pad_width=1, mode='constant')
    pc0 = pixel_coords[0] + 1
    pc1 = pixel_coords[1] + 1  # plus one bc of padding
    prop_to = []

    if is_diagonal_a_neighbor:
        candidate_coords_list = ([(pc0-1, pc1-1), (pc0-1, pc1), (pc0-1, pc1+1),
                                  (pc0, pc1-1),                (pc0, pc1+1),
                                  (pc0+1, pc1-1), (pc0+1, pc1), (pc0+1, pc1+1)])
    else:
        candidate_coords_list = ([(pc0-1, pc1),
                                  (pc0, pc1-1),                (pc0, pc1+1),
                                  (pc0+1, pc1)])

    for cc in candidate_coords_list:
        if padded_can_arr[cc] > 0:  # pixel corresponds to a canal
            if padded_dem[cc] - padded_dem[pc0, pc1] > -threshold:
                prop_to.append(int(padded_can_arr[cc]))
    return prop_to


def label_canal_pixels(can_arr, dem):
    # Convert labels of canals to 1,2,3...
    aux = can_arr.flatten()
    can_flat_arr = np.array(aux)
    aux_dem = dem.flatten()

    counter = 1
    for i, value in enumerate(aux):
        # if there is a canal in that pixel and if we have a dem value for that pixel. (Missing dem data points are labelled as -9999)
        if value > 0 and aux_dem[i] > 0:
            can_flat_arr[i] = counter
            counter += 1
    # contains labels and positions of canals
    labelled_canals = can_flat_arr.reshape(can_arr.shape)

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
    # lil matrix is good to build it incrementally
    matrix = scipy.sparse.lil_matrix((n_canals, n_canals))

    c_to_r_list = [0] * n_canals
    for coords, label in np.ndenumerate(labelled_canals):
        if labelled_canals[coords] > 0:  # if coords correspond to a canal.
            # label=0 is not a canal. But do not delete it, otherwise everything would be corrido.
            c_to_r_list[int(label)] = coords
            propagated_to = _prop_to_neighbours(
                coords, labelled_canals, dem, threshold=0.0, is_diagonal_a_neighbor=False)
            for i in propagated_to:
                # adjacency matrix of the directed graph
                matrix[int(label), i] = 1

    matrix_csr = matrix.tocsr()  # compressed row format is more efficient for later
    matrix_csr.eliminate_zeros()  # happens in place. Frees disk usage.

    return matrix_csr, c_to_r_list


def get_array_indices_by_rows(array_shape):
    rows, cols = np.indices(array_shape)
    indices = np.array(list(zip(rows.flatten(), cols.flatten())))
    final_shape = list(array_shape) + [2]  # The 2 comes from the 2D

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


def get_lists_of_node_coords_and_node_heights(raster_src, lines_gpkg):
    # get all unique line edpoints
    nodes_coords = cwl_utilities.get_unique_nodes_in_line_gpkg(lines_gpkg)
    
    nodes_heights = []
    nodes_coords_copy = nodes_coords.copy()
    for i, dem_height in enumerate(raster_src.sample(nodes_coords)):
        if np.isnan(dem_height[0]) or dem_height[0] < 0:  # Coordinates are outside the DEM
            # Remove from list of nodes
            nodes_coords_copy.remove(nodes_coords[i])
        else:
            nodes_heights.append(dem_height[0])
    nodes_coords = nodes_coords_copy.copy()
    del nodes_coords_copy

    return nodes_coords, nodes_heights


def build_coords_to_height_dict(nodes_coords, nodes_heights):
    return {coords: h for coords, h in zip(nodes_coords, nodes_heights)}


def build_number_nodes_to_coord_dict(nodes_coords):
    return {n: (coord_x, coord_y) for n, (coord_x, coord_y) in enumerate(nodes_coords)}


def build_coords_to_number_nodes_dict(nodes_coords):
    return {(coord_x, coord_y): n for n, (coord_x, coord_y) in enumerate(nodes_coords)}


def create_graph_from_edges_with_node_attributes(edges, nodes_attribute_dict):
    graph = nx.DiGraph()
    graph.add_edges_from(edges)
    graph.add_nodes_from(nodes_attribute_dict)
    return graph

def create_geodataframe_with_simplified_straight_lines(geodf):
    # Takes a geodataframe with LineStrings as geometries and selects only
    # the beginning and end points of the LineStrings to make a new, simplified, straight
    # Linestring object. Then,
    # returns a new gdf with those simple lines
    geometries = []
    for index, row in geodf.iterrows():
        coords = [(coords) for coords in list(row['geometry'].coords)]
        first_coord, last_coord = [ coords[i] for i in (0, -1) ]
        geometries.append(LineString([Point(first_coord), Point(last_coord)]))
    
    return geopandas.GeoDataFrame(geometry=geometries) 

def read_lines_and_raster_and_produce_dirty_graph(gpkg_fn, fn_dtm, edge_direction_is_local=False, simplify_straighten_lines=True):
    lines_gpkg = geopandas.read_file(gpkg_fn)
    if simplify_straighten_lines:
        lines_gpkg = create_geodataframe_with_simplified_straight_lines(lines_gpkg)
    raster_src = rasterio.open(fn_dtm)
    nodes_coords, nodes_heights = get_lists_of_node_coords_and_node_heights(
        raster_src, lines_gpkg)

    dict_coords_to_height_nodes = build_coords_to_height_dict(
        nodes_coords, nodes_heights)

    dict_coords_to_number_nodes = build_coords_to_number_nodes_dict(
        nodes_coords)
    if edge_direction_is_local:
        # Edge direction from height difference between neighbors
        edges = cwl_utilities.get_slope_directed_edges_with_node_number_name(
            lines_gpkg, dict_coords_to_number_nodes, dict_coords_to_height_nodes)
    else:
        # Edge direction inherited from file
        edges = []
        for line in lines_gpkg.geometry:
            try:
                up_node_number = dict_coords_to_number_nodes[line.coords[0]]
                down_node_number = dict_coords_to_number_nodes[line.coords[-1]]
                edges.append((up_node_number, down_node_number))
            except: # canal node outside of dem
                continue
        
    nodes_attribute_dict = [(n, {"x": coord_x, "y": coord_y, 'DEM': dict_coords_to_height_nodes[(coord_x, coord_y)]})
                            for n, (coord_x, coord_y) in enumerate(nodes_coords)]

    graph = create_graph_from_edges_with_node_attributes(
        edges, nodes_attribute_dict)

    return graph


def get_duplicated_up_and_down_junction_nodes(all_junctions):
    # Returns nodes that are downstream and upstream nodes in more than 1 junction
    all_up_junction_nodes = [up_node for up_nodes,
                             _ in all_junctions for up_node in up_nodes]
    all_down_junction_nodes = [
        down_node for _, down_nodes in all_junctions for down_node in down_nodes]

    up_dupli = list(cwl_utilities.get_duplicates_in_list(
        all_up_junction_nodes))
    down_dupli = list(cwl_utilities.get_duplicates_in_list(
        all_down_junction_nodes))

    return up_dupli, down_dupli


def remove_edge_between_nodes(adj_mat, source_node, target_node):
    # source_node is where the (directed!) edge begins and target_node is where it ends
    adj_changed = adj_mat.copy()
    if adj_changed[target_node, source_node] == 0:
        raise ValueError('The edge you want to remove does not exist')
    else:
        adj_changed[target_node, source_node] = 0

    return adj_changed


def does_cnm_have_bad_junctions(down_junction_duplicated):
    if len(down_junction_duplicated) == 0:
        return False
    else:
        return True


def remove_one_bad_edge(canal_network_matrix, up_junction_duplicated, down_junction_duplicated):
    # Remove bad junctions, one edge at a time
    cnm = canal_network_matrix.copy()

    down_dupli = down_junction_duplicated[0]  # take only the first one
    # predecessors of down_dupli
    predecessors = list(np.where(cnm[down_dupli])[0])
    predecessors_that_are_up_junctions = [
        p for p in predecessors if p in up_junction_duplicated]
    if len(predecessors_that_are_up_junctions) > 0:
        cnm = remove_edge_between_nodes(
            cnm, predecessors_that_are_up_junctions[0], down_dupli)
    else:
        raise ValueError(
            'None of the predecessors appear in more than one junction!')

    return cnm


def clean_cnm_from_strange_junctions(cnm):
    """Removes faulty multiple count of junctions in the 
    canal network matrix. By "faulty" here I mean the case where
    one node is the up- or downstream node of more than one junction.
    To do that, it simply removes annoying edges from the cnm. 

    Args:
        cnm (numpy.ndarray): Canal network matrix

    Returns:
        cnm (numpy.ndarray): Canal network matrix without faulty junctions
    """
    cnm_is_dirty = True
    cnm_clean = cnm.copy()
    while cnm_is_dirty:
        junctions = cwl_utilities.get_all_junctions(cnm_clean)
        up_junction_duplicated, down_junction_duplicated = get_duplicated_up_and_down_junction_nodes(
            junctions)
        if does_cnm_have_bad_junctions(down_junction_duplicated):
            try:
                cnm_clean = remove_one_bad_edge(
                    cnm_clean, up_junction_duplicated, down_junction_duplicated)
            except:
                print('could not remove edge')
        else:
            cnm_is_dirty = False

    return cnm_clean


def clean_graph_from_strange_junctions(graph):
    """Removes faulty multiple count of junctions in the 
    canal network matrix. By "faulty" here I mean the case where
    one node is the up- or downstream node of more than one junction.
    To do that, it simply removes annoying edges from the cnm. 

    Args:
        graph (networkx.DiGraph): Graph of the canal network matrix

    Returns:
        graph (networkx.DiGraph): Graph of the canal network matrix without faulty junctions
    """

    graph_is_dirty = True
    nodelist = list(range(0, len(graph.nodes)))
    while graph_is_dirty:
        # nodelist forces the adjacency matrix to have an order.
        cnm = nx.to_numpy_array(graph, nodelist).T
        junctions = cwl_utilities.get_all_junctions(cnm)
        up_junction_duplicated, down_junction_duplicated = get_duplicated_up_and_down_junction_nodes(
            junctions)
        if does_cnm_have_bad_junctions(down_junction_duplicated):
            try:  # Remove one bad edge
                # take only the first one
                down_dupli = down_junction_duplicated[0]
                # predecessors of down_dupli
                predecessors = list(np.where(cnm[down_dupli])[0])
                predecessors_that_are_up_junctions = [
                    p for p in predecessors if p in up_junction_duplicated]
                edge_to_remove = (
                    predecessors_that_are_up_junctions[0], down_dupli)
                graph.remove_edge(*edge_to_remove)
            except:
                raise ValueError('could not remove edge')

        else:
            graph_is_dirty = False

    return graph


def find_first_match_in_two_lists(list1: list, list2: list):
    """Finds the first occurrence of an equal element between two lists.
    It avoids iterating over the whole lists just to get a match.
    Args:
        list1 (list): A list
        list2 (list): A list

    Returns:
        [type] or None: If an element exists in both lists, then the first such element is returned.
                        If not, then None is returned.
    """
    return next((i for i in list1 if i in set(list2)), None)


def get_junction_up_or_downsteam_nodes(junctions, mode):
    # mode = 'up' or 'down'
    nodes = []
    if mode == 'down':
        for _, downs in junctions:
            nodes = nodes + downs
        return nodes
    elif mode == 'up':
        for ups, _ in junctions:
            nodes = nodes + ups
        return nodes
    else:
        raise ValueError('mode should be either "up" or "down"')
        return None


def clean_graph_from_spy_nodes(graph, mode):
    # Spy nodes are nodes that are simultaneously either
    # downstream BCs AND downstream junction nodes
    # OR
    # upstream BCs AND upstream junction nodes
    # mode can be 'delete_all' or 'delete_only_necessary', depending on
    # whether all spy nodes are deleted at once, or they are deleted in a loop one at a time
    graph_is_dirty = True
    nodelist = list(range(0, len(graph.nodes)))
    while graph_is_dirty:
        # nodelist forces the adjacency matrix to have an order.
        cnm = nx.to_numpy_array(graph, nodelist).T

        junctions = cwl_utilities.get_all_junctions(cnm)
        junction_downs = get_junction_up_or_downsteam_nodes(
            junctions, mode='down')
        junction_ups = get_junction_up_or_downsteam_nodes(junctions, mode='up')

        upstream_bool, downstream_bool = cwl_utilities.infer_extreme_nodes(
            cnm)  # BC nodes
        downstream_nodes = tuple(np.argwhere(downstream_bool).flatten())
        upstream_nodes = tuple(np.argwhere(upstream_bool).flatten())

        if mode == 'delete_only_necessary':
            down_spy_node = find_first_match_in_two_lists(
                downstream_nodes, junction_downs)
            up_spy_node = find_first_match_in_two_lists(
                upstream_nodes, junction_ups)

            if down_spy_node is not None:
                predecessor = list(np.where(cnm[down_spy_node])[0])
                edge_to_remove = (predecessor[0], down_spy_node)
                graph.remove_edge(*edge_to_remove)
            if up_spy_node is not None:
                succesor = list(np.where(cnm[:, up_spy_node])[0])
                edge_to_remove = (up_spy_node, succesor[0])
                graph.remove_edge(*edge_to_remove)
            else:
                graph_is_dirty = False

        elif mode == 'delete_all':
            down_spy_nodes = set(
                [n for n in downstream_nodes if n in junction_downs])
            predecessors_of_down = [list(graph.predecessors(n))[
                0] for n in down_spy_nodes]

            up_spy_nodes = set(
                [n for n in upstream_nodes if n in junction_ups])
            successors_of_up = [list(graph.successors(n))[0]
                                for n in up_spy_nodes]

            edges_to_remove = [(u, d) for d, u in zip(
                down_spy_nodes, predecessors_of_down)]
            edges_to_remove = edges_to_remove + \
                [(u, d) for u, d in zip(up_spy_nodes, successors_of_up)]
            edges_to_remove = set(edges_to_remove)

            if len(edges_to_remove) < 1:
                graph_is_dirty = False

            for e in edges_to_remove:
                try:
                    graph.remove_edge(*e)
                except:
                    print('edge not found. Could not be removed')

        else:
            return ValueError('mode not understood')

    return graph


def compute_channel_network_graph(gpkg_fn, fn_dtm):
    # This takes a long time! (approx. 1hour)
    # 1st, clean from streange junctions, then clean from spy nodes
    graph = read_lines_and_raster_and_produce_dirty_graph(
        gpkg_fn, fn_dtm)
    graph_clean = clean_graph_from_strange_junctions(graph)
    graph_clean = clean_graph_from_spy_nodes(
        graph_clean.copy(), mode='delete_all')

    return graph_clean.copy()


def load_graph(load_from_pickled=True):
    if load_from_pickled:
        graph = pickle.load(open("canal_network_matrix.p", "rb"))
    else:
        # Windows
        gpkg_fn = r"C:\Users\03125327\github\canal_net\qgis\final_lines.gpkg"
        fn_dtm = r"C:\Users\03125327\Documents\qgis\canal_networks\dtm_10x10.tif"
        # Linux -own computer-
        #gpkg_fn = r"/home/txart/Programming/GitHub/canal_net/qgis/final_lines.gpkg"
        #fn_dtm10x10 = r"/home/txart/Programming/data/dtm_10x10.tif"

        graph = compute_channel_network_graph(gpkg_fn, fn_dtm)

    return graph

# %%

# %%
