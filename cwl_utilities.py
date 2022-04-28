import numpy as np
import pandas as pd
import geopandas
from numba import njit
from numpy.core.fromnumeric import nonzero
import networkx as nx
import rasterio


def merge_two_dictionaries(dict1, dict2):
    res = {**dict1, **dict2}
    return res

def infer_extreme_nodes(directed_adj_matrix):
    """
    Infer what nodes are the beginning and end of canals from adjacency matrix.
    Last nodes of canals  are identified by having no outgoing edges
    First nodes of canals have no incoming edges

    Parameters
    ----------
    adj_matrix : numpy array
        Adjacency matrix of the canal network graph

    Returns
    -------
    end_nodes_bool : boolean numpy array
        True where nodes are last nodes of canals
    first_nodes_bool : boolean numpy array
        True where nodes are first nodes of canals

    """
    # raise NotImplemented('Deprecated in 14.1.2022. Now using infer_extreme_nodes_with_networkx(digraph)')
    # Infer neumann and Diri nodes from adj matrix
    # Boundary values below are conditional on this boolean mask
    end_nodes_bool = np.sum(directed_adj_matrix, axis=0) == 0
    first_nodes_bool = np.sum(directed_adj_matrix, axis=1) == 0
    # in case the summing over the sparse matrix changes the numpy array shape
    end_nodes_bool = np.ravel(end_nodes_bool)
    first_nodes_bool = np.ravel(first_nodes_bool)

    return first_nodes_bool, end_nodes_bool

def infer_extreme_nodes_with_networkx(digraph):
    upstream_nodes = []; downstream_nodes = []
    for node in list(digraph.nodes):
        if digraph.out_degree[node] == 0:
            downstream_nodes.append(node)
        elif digraph.in_degree[node] == 0:
            upstream_nodes.append(node)

    return upstream_nodes, downstream_nodes            

def choose_graph(graph_name, n_nodes=5):
    """
    A small library of pre-defined networks.
    """

    if graph_name == 'line':
        graph = np.diag(np.ones(n_nodes-1), k=-1)

    elif graph_name == 'Y':
        nn = int(n_nodes/3)  # nodes per branch

        graph = np.block([
            [np.diag(np.ones(nn-1), k=-1), np.zeros((nn, nn)),
             np.zeros((nn, nn+1))],
            [np.zeros((nn, nn)), np.diag(np.ones(nn-1), k=-1),
             np.zeros((nn, nn+1))],
            [np.zeros((nn+1, nn)), np.zeros((nn+1, nn)), np.diag(np.ones(nn), k=-1)]])
        graph[2*nn, nn-1] = 1
        graph[2*nn, 2*nn-1] = 1

    elif graph_name == 'tent':
        if n_nodes % 2 == 0:
            raise ValueError('number of nodes has to be odd for tent')
        hn = int(n_nodes/2)
        graph = np.block([
                         [np.diag(np.ones(hn-1), k=1), np.zeros((hn, hn))],
                         [np.zeros((hn, hn)), np.diag(np.ones(hn-1), k=-1)]
                         ])
        graph = np.insert(arr=graph, obj=hn,
                          values=np.zeros(n_nodes-1), axis=0)
        graph = np.insert(arr=graph, obj=hn, values=np.zeros(n_nodes), axis=1)
        graph[hn-1, hn] = 1
        graph[hn+1, hn] = 1

    elif graph_name == 'ring':
        graph = np.array([[0, 0, 0, 0, 1],
                         [1, 0, 0, 0, 0],
                         [0, 1, 0, 0, 0],
                         [0, 0, 1, 0, 0],
                         [0, 0, 0, 1, 0]])

    elif graph_name == 'lollipop':
        graph = np.array([[0, 0, 0, 0, 0],
                          [1, 0, 0, 0, 0],
                          [1, 0, 0, 0, 0],
                          [0, 1, 1, 0, 0],
                          [0, 0, 0, 1, 0]])

    elif graph_name == 'grid':
        graph = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0]])
    return graph


def choose_bottom(graph_name, n_nodes):
    if graph_name == 'tent':
        bottom = np.hstack((np.linspace(1, 3, int(n_nodes/2)),
                            np.linspace(3, 1, int(n_nodes/2))))
        bottom = np.insert(bottom, int(n_nodes/2), 3.05)

    elif graph_name == 'Y':
        nodes_per_branch = int(n_nodes/3)
        branch_1_top = 3.1
        branch_2_top = 3.2
        junction = 3
        branch_3_top = junction - 0.4
        branch_3_end = 1
        top_branch = np.linspace(start=branch_1_top,
                                 stop=junction, num=nodes_per_branch)
        branch_2 = np.linspace(start=branch_2_top,
                               stop=junction, num=nodes_per_branch)
        branch_3 = np.linspace(start=branch_3_top,
                               stop=branch_3_end, num=nodes_per_branch+1)

        bottom = np.hstack((top_branch, np.hstack((branch_2, branch_3))))

    else:
        # default downward slope
        bottom = np.linspace(start=5, stop=3,  num=n_nodes)

    return bottom


def read_true_data():
    import pandas as pd
    from pathlib import Path
    import preprocess_data

    filenames_df = pd.read_excel('file_pointers.xlsx', header=2, dtype=str, engine='openpyxl')

    dem_rst_fn = Path(
        filenames_df[filenames_df.Content == 'DEM'].Path.values[0])
    can_rst_fn = Path(
        filenames_df[filenames_df.Content == 'canal_raster'].Path.values[0])

    # Choose smaller study area
    # E.g., a study area of (0,-1), (0,-1) is the whole domain
    STUDY_AREA = (0, -1), (0, -1)

    can_arr, wtd_old, dem, _, peat_depth_arr, _, _ = preprocess_data.read_preprocess_rasters(
        STUDY_AREA, dem_rst_fn, can_rst_fn, dem_rst_fn, dem_rst_fn, dem_rst_fn, dem_rst_fn, dem_rst_fn)
    labelled_canals = preprocess_data.label_canal_pixels(can_arr, dem)
    CNM, c_to_r_list = preprocess_data.gen_can_matrix_and_label_map(
        labelled_canals, dem)
    bottom = [dem[loc] for loc in c_to_r_list]
    bottom[0] = 3.0  # something strange happens with this node
    bottom = np.array(bottom)

    c_to_r_list.pop(0)  # first is useless for some reason

    return CNM, bottom, c_to_r_list

def get_duplicates_in_list(input_list):
    """Finds duplicates in a Python list and returns a set of them.
    NOTE: it is faster than using the built-in count() method because
    count() parses the whole list in each iteration, whereas this method
    only checks if element has been seen before.

    Args:
        input_list (list): list where duplicates are looked for

    Returns:
        set: set of duplicates in input_list
    """     
    seen_values = set()
    duplicate_values = set()
    for value in input_list:
        if value not in seen_values:
            seen_values.add(value)
        else:
            duplicate_values.add(value)
            
    return duplicate_values

def find_downstream_or_upstream_junction_nodes(A, down_or_up):
    """given the canal network adj matrix, finds junctions and creates
    dict of {[upstream nodes] : [downstream nodes]}

    Args:
        A (numpy array): adj matrix of canal network
        down_or_up (str): options: 'down' or 'up'. 'Down' finds junctions
        with multiple downstream nodes. 'Up' does the upstreams.
    """
    if type(A) != np.ndarray:
        raise TypeError(
            "The matrix must be a numpy array. In particular, scipy sparse matrices work badly with this function.")

    if down_or_up != 'down' and down_or_up != 'up':
        raise ValueError('down_or_up has to be either "down" or "up"')

    axis_to_sum = 0 if down_or_up == 'down' else 1

    junctions = []
    all_down_nodes = np.argwhere(np.sum(A, axis=axis_to_sum) > 1).flatten()
    for down_node in all_down_nodes:
        if down_or_up == 'up':
            up_nodes = np.argwhere(A[down_node]).flatten().tolist()
            junctions.append([up_nodes, [down_node]])
        else:
            # here down means up and up, down
            up_nodes = np.argwhere(A[:, down_node]).flatten().tolist()
            junctions.append([[down_node], up_nodes])

    return junctions

def sparse_get_all_junctions(A):
    # This function works with sparse input arrays A, unlike the 
    # numpy array variant "get_all_junctions()"  
    nonzero_row_indices, nonzero_col_indices = A.nonzero()
    
    down_junction_nodes = get_duplicates_in_list(list(nonzero_row_indices))
    up_junction_nodes = get_duplicates_in_list(list(nonzero_col_indices))
    
    junctions = []
    for up in up_junction_nodes:
        downs = list(nonzero_row_indices[np.where(nonzero_col_indices==up)[0]])
        junctions.append([[up], downs])
    for down in down_junction_nodes:
        ups = list(nonzero_col_indices[np.where(nonzero_row_indices==down)[0]])
        junctions.append([ups, [down]])
    
    return junctions
        


def get_all_junctions(cnm):
    junctions_with_multiple_up = find_downstream_or_upstream_junction_nodes(
            cnm, down_or_up='up')
    junctions_with_multiple_down = find_downstream_or_upstream_junction_nodes(
        cnm, down_or_up='down')
    return junctions_with_multiple_down + junctions_with_multiple_up

def get_all_junctions_with_networkx(digraph):
    # NetworkX implementation of get_all_junctions
    junctions = []
    for node in list(digraph.nodes):
        if digraph.out_degree[node] > 1: # it's an "up junction node"
            downs = list(digraph.neighbors(node))
            junctions.append([[node], downs])
            
        elif digraph.in_degree[node] > 1: # it's a down junction node
            ups = list(digraph.predecessors(node))
            junctions.append([ups, [node]])

    return junctions


def convert_list_of_junctions_to_numpy_arr(list_of_junctions):
    MAX_JUNCTION_SIZE = 10  # cutoff for max down or upstream nodes in a junction
    array = np.zeros(shape=(len(list_of_junctions),
                     2, MAX_JUNCTION_SIZE), dtype=int)
    for i in range(len(list_of_junctions)):
        array[i, 0] = np.pad(list_of_junctions[i][0], pad_width=(
            0, MAX_JUNCTION_SIZE-len(list_of_junctions[i][0])))
        array[i, 1] = np.pad(list_of_junctions[i][1], pad_width=(
            0, MAX_JUNCTION_SIZE-len(list_of_junctions[i][1])))

    return array


# Tools to build the jacobian


def tridiag(a, b, c, k1=-1, k2=0, k3=1):
    return np.diag(a, k1) + np.diag(b, k2) + np.diag(c, k3)


@njit
def interweave_vectors(v1, v2):
    """
    Takes 2 vectors of the same dimension, [a1, a2, ...] and [b1, b2, ...]
    and returns [a1, b1, a2, b2, ...].

    """
    if v1.shape != v2.shape:
        raise ValueError('arrays must have the same shape')

    shape = 2*v1.shape[0]
    result = np.zeros(shape)
    result[0::2] = v1
    result[1::2] = v2
    return result


@njit
def interweave_matrices(array1, array2):
    """
    Takes 2 matrices of the same dimension, [[a11, a12, ...],...] and [[b11, b12, ...], ...]
    and returns an alternatemaatrix by rows [[a11, a12, ...],
                 [b11, b12, ...],
                 ...].

    """
    if array1.shape != array2.shape:
        raise ValueError('arrays must have the same shape')

    shape = (2*array1.shape[0], array1.shape[1])
    result = np.zeros(shape)
    result[0::2] = array1
    result[1::2] = array2
    return result


@njit
def deinterweave_array(array):
    """
    The inversee of interweave_arrays.
    Takes an interweaved array and returns 2 arrays.
    """
    if array.shape[0] % 2 != 0:
        raise ValueError('the 0th dimension of the array must be even')

    return array[0::2], array[1::2]


@njit
def pad_array_by_interweaving(array, position_zeroes=('right', 'bottom')):
    """Returns an array of double the size of the input array
    with a zero between every entry of the array

    Args:
        array (numpy array): input, square array
        position__zeroes (tuple(bool)): Specifies where to pad with the zeroes.
        Options for the first position aare 'left'and 'right'. For the second,
        'top'and 'bottom'.

    """
    if position_zeroes[1] == 'top':
        temp = interweave_matrices(np.zeros_like(array), array)
    elif position_zeroes[1] == 'bottom':
        temp = interweave_matrices(array, np.zeros_like(array))

    if position_zeroes[0] == 'left':
        return interweave_matrices(np.zeros_like(temp.T), temp.T).T
    elif position_zeroes[0] == 'right':
        return interweave_matrices(temp.T, np.zeros_like(temp.T)).T



#%% Read shapefile type of data for the channel network

def get_slope_directed_edges_with_node_number_name(lines_gpkg, dict_coords_to_number_nodes, dict_coords_to_height_nodes):
    edges = []
    for line in lines_gpkg.geometry:
        try:
            height0 = dict_coords_to_height_nodes[(line.coords[0])]
            height1 = dict_coords_to_height_nodes[(line.coords[1])]
        except: # the line ends are not in the dictionaries. For instance, because they are outside the DEM.
            continue
        if height0 >= height1:  # edge from 0 to 1
            coords0 = line.coords[0]
            coords1 = line.coords[1]
        else:  # edge from 1 to 0
            coords0 = line.coords[1]
            coords1 = line.coords[0]

        edges.append(
            (dict_coords_to_number_nodes[coords0], dict_coords_to_number_nodes[coords1]))
    return edges


def get_slope_directed_edges_with_node_number_name_MULTILINESTRING(lines_gpkg, dict_coords_to_number_nodes, dict_coords_to_height_nodes):
    # VERSION OF THE ABOVE WITH EXTRAA LOOP FOR MULTILINESTRING OBJECTS IN GPKG.
    edges = []
    for geom in lines_gpkg.geometry:
        for line in geom: # This extra loop is necessary when geometry is a MULTILINESTRING instead of a LINESTRING
            try:
                height0 = dict_coords_to_height_nodes[(line.coords[0])]
                height1 = dict_coords_to_height_nodes[(line.coords[1])]
            except: # the line ends are not in the dictionaries. For instance, because they are outside the DEM.
                continue
            if height0 >= height1:  # edge from 0 to 1
                coords0 = line.coords[0]
                coords1 = line.coords[1]
            else:  # edge from 1 to 0
                coords0 = line.coords[1]
                coords1 = line.coords[0]

        edges.append(
            (dict_coords_to_number_nodes[coords0], dict_coords_to_number_nodes[coords1]))
    return edges

def get_all_line_points_from_geopackage_file(lines_gpkg):
    """Take a read geopackage file containing only lines 
    and spit out a list of all the points of those lines.
    NOTE: this could have also been done with the nodes geopackage file

    Args:
        lines_gpkg (geopandas geodataframe): holds the output of  geopandas.readfile("*.gpkg")

    Returns:
        (list): list of ends of lines in the geopackage file
    """
    all_line_points = []
    for line in lines_gpkg.geometry:
        for n1, n2 in line.coords:
            all_line_points.append((n1, n2))

    return all_line_points


def get_all_line_points_from_geopackage_file_MULTILINESTRING(lines_gpkg):
    """ VERSION OF THE ABOVE FOR  GEOMETRIES WITH MULTILINESTRING
    Take a read geopackage file containing only lines 
    and spit out a list of all the points of those lines.
    NOTE: this could have also been done with the nodes geopackage file

    Args:
        lines_gpkg (geopandas geodataframe): holds the output of  geopandas.readfile("*.gpkg")

    Returns:
        (list): list of ends of lines in the geopackage file
    """
    all_line_points = []
    for geom in lines_gpkg.geometry:
        for line in geom:  # This extra loop is necessary when geometry is a MULTILINESTRING instead of a LINESTRING
            for n1, n2 in line.coords:
                all_line_points.append((n1, n2))

    return all_line_points


def get_all_line_endpoints_from_geopackage_file(lines_gpkg):
    """Take a read geopackage file containing only lines 
    and spit out a list of the first and last points (endpoints) of that line.

    Args:
        lines_gpkg (geopandas geodataframe): holds the output of  geopandas.readfile("*.gpkg")

    Returns:
        (list): list of ends of lines in the geopackage file
    """
    all_line_endpoints = []
    for line in lines_gpkg.geometry:
        try:
            n1 = line.coords[0]
            n2 = line.coords[-1]
            all_line_endpoints.append(n1)
            all_line_endpoints.append(n2)
        except:
            raise Warning(" There was some problem with a line when trying to get its coords")
            continue

    return all_line_endpoints

def get_unique_nodes_in_line_gpkg(lines_gpkg):
    # nodes are the unique values of all_line_endpoints
    return list(set(get_all_line_endpoints_from_geopackage_file(lines_gpkg)))




#%% Reverse Cuthill-Mckee permutation
def permute_row_columns(permutation_vector, array):
    permuted_array = array.copy()
    permuted_array = permuted_array[permutation_vector, :]
    permuted_array = permuted_array[:, permutation_vector]

    return permuted_array


def permute_vector(permutation_vector, vector):
    # Example: if perm vector is [0,3,4,2]
    # It performs the permutation [a,b,c,d] -> [a, c, d, b]
    return vector[permutation_vector]

def convert_networkx_graph_to_dataframe(graph):
    dict_nodes_to_attributes = dict(graph.nodes.data(True))
    df = pd.DataFrame.from_dict(dict_nodes_to_attributes, orient='index')

    return df

def create_dataframe_from_node_attributes(graph) -> pd.DataFrame:
    df_nodes = convert_networkx_graph_to_dataframe(graph)
    df_nodes = df_nodes.reset_index().rename(columns={'index': 'n_node'})

    return df_nodes

def convert_dataframe_to_geodataframe(dataframe: pd.DataFrame, name_of_x_column: str, name_of_y_column: str, projection='epsg:32748') -> geopandas.GeoDataFrame:
    """Geodataframes by definition contain spatially explicit information.
    For this function to work, there must be 2 columns is the input dataframe which hold
    longitude and latitude values. The names of these columns are specified under
    name_of_x_column and name_of_y_column.
    Also, it is necessary to specify a projection (or coordinate reference system name). 
    The rest of the columns of the dataframe pass directly to the geodataframe

    Args:
        dataframe (pd.DataFrame): dataframe with longitude and latitude values
        name_of_x_column (str): name of the column in the dataframe that holds long values
        name_of_y_column (str): name of the column in the dataframe that holds lat values
        projection (str): the coordinate reference system to use. By default, 'epsg:32748'

    Returns:
        geopandas.GeoDataFrame: A geodataframe containing all the data from the input dataframe
    """

    projection = 'epsg:32748'
    gdf = geopandas.GeoDataFrame(dataframe, geometry=geopandas.points_from_xy(
        dataframe[name_of_x_column], dataframe[name_of_y_column]), crs=projection)

    return gdf

def subtract_dictionary_values(d1, d2):
    # d1 - d2. Loops over d1's keys.
    return {key: d1[key] - d2.get(key, 0) for key in d1}



