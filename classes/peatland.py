import pandas as pd
from pathlib import Path
import networkx as nx
import numpy as np
import rasterio
import platform

from classes.channel_network import ChannelNetwork
import preprocess_data
import utilities

class Peatland:
    def __init__(self, cn: ChannelNetwork, fn_pointers) -> None:
        
        # filenames
        filenames_df = pd.read_excel(fn_pointers, header=2, dtype=str, engine='openpyxl')
        self.fn_dem = Path(filenames_df[filenames_df.Content == 'DEM'].Path.values[0])
        self.fn_depth = Path(filenames_df[filenames_df.Content == 'depth'].Path.values[0])
        # fn_peat_type = filenames_df[filenames_df.Content == 'peat_type_raster'].Path.values[0]
        self.fn_weather_station_locations = filenames_df[filenames_df.Content == 'weather_station_coords'].Path.values[0]
        self.fn_canal_mask_mesh_centroids = filenames_df[filenames_df.Content == 'canal_mask_mesh_centroids'].Path.values[0]
        self.fn_template_output_raster =   filenames_df[filenames_df.Content == 'template_output_raster'].Path.values[0]

        
        # Arrays
        self.dem = self.read_and_preprocess_dem(self.fn_dem)
        self.peat_depth = self.read_and_preprocess_peat_depth(self.fn_depth)
        self.template_output_raster = self.read_and_preprocess_dem(self.fn_template_output_raster)


        # Quantities derived from arrays
        self.catchment_mask = self.dem >= 0

        # Channel network stuff
        # Set the position of each node in the underlying DEM raster
        dict_node_to_pos_in_dem = self.map_node_number_to_pixel_pos_in_raster(
            cn.graph, self.fn_dem)
        # store information in graph
        nx.set_node_attributes(
            cn.graph, values=dict_node_to_pos_in_dem, name='pos_in_raster')

        # Get the position of each node in the underlying raster
        self.dict_node_to_pos_in_raster = self.map_node_number_to_pixel_pos_in_raster(
            cn.graph, self.fn_dem)
        self.dict_pos_in_raster_to_node_number = utilities.invert_dictionary(
            self.dict_node_to_pos_in_raster, are_values_unique=False)

        # # canal mask raster
        # dem_in_canals = self.rasterize_graph_attribute_to_dem_shape(
        #     graph=cn.graph, graph_attr='DEM')
        # canal_mask = dem_in_canals != 0

        # # eliminate potential canals outside catchment
        # intersection_of_bcs = canal_mask*(~self.catchment_mask)
        # canal_mask[intersection_of_bcs] = False

        # self.canal_mask = canal_mask

        pass

    def read_and_preprocess_dem(self, fn_dem):
        dem = preprocess_data.read_raster(fn_dem)
        OUT_OF_BOUNDS_DEM_VALUE = -9999 # Negative value for catchment mask, etc. Later changed to positive.
        dem[dem < -10] = OUT_OF_BOUNDS_DEM_VALUE
        dem[np.where(np.isnan(dem))] = OUT_OF_BOUNDS_DEM_VALUE
        dem[dem > 1e20] = OUT_OF_BOUNDS_DEM_VALUE  # just in case
        return dem

    def read_and_preprocess_peat_depth(self, fn_peat_depth):
        peat_depth_arr = preprocess_data.read_raster(fn_peat_depth)
        # peat_depth_arr[peat_depth_arr < 0] = -1
        return peat_depth_arr

    def peat_depth_map(self, peat_depth_type_arr):
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

    def peat_depth_from_peat_type(self, fn_peat_type):
        peat_depth_arr = preprocess_data.read_raster(fn_peat_type)
        peat_depth_arr = self.peat_depth_map(peat_depth_arr)  # translate number keys to depths
        # fill some nodata values to get same size as dem
        peat_depth_arr[(np.where(self.dem > 0.1)
                        and np.where(peat_depth_arr < 0.1))] = 1.
        return peat_depth_arr

    def read_and_preprocess_peat_type(self, fn_peat_type):
        peat_type_arr = preprocess_data.read_raster(fn_peat_type)
        peat_type_arr[peat_type_arr < 0] = -1
        # fill some nodata values to get same size as dem
        peat_type_arr[(np.where(self.dem > 0.1)
                       and np.where(peat_type_arr < 0.1))] = 1.
        return peat_type_arr

    def map_node_number_to_pixel_pos_in_raster(self, graph, fn_raster):
        dict_node_number_pos_dem_array = {}
        with rasterio.open(fn_raster) as raster:
            for n, attr in graph.nodes(data=True):
                px, py = raster.index(attr['x'], attr['y'])
                dict_node_number_pos_dem_array[n] = (px, py)

        return dict_node_number_pos_dem_array

    def rasterize_graph_attribute_to_dem_shape(self, graph, graph_attr):
        output_raster = np.zeros(shape=self.dem.shape)
        for arr_pos, n_nodes in self.dict_pos_in_raster_to_node_number.items():
            sum_of_values = 0
            for n in n_nodes:
                sum_of_values += graph.nodes[n][graph_attr]
            # Raster value is mean of all values
            output_raster[arr_pos] = sum_of_values / len(n_nodes)

        return output_raster

    def set_raster_value_to_graph_attribute(self, raster_value, graph, graph_attr: str):
        for n_node, arr_pos in self.dict_node_to_pos_in_raster.items():
            graph.nodes(data=True)[n_node][graph_attr] = raster_value[arr_pos]
        return None
    
    def read_weather_station_locations(self):
        df_weather_station_coords = pd.read_excel(self.fn_weather_station_locations, engine='openpyxl')
        weather_station_coords_dict = {}
        for i, row in df_weather_station_coords.iterrows():
            weather_station_coords_dict[row.Name] = np.array([row.X, row.Y])
        
        # This same order is used for other computations in peatland hydro
        weather_station_order_in_array = list(df_weather_station_coords['Name'])
            
        return np.array([weather_station_coords_dict[wname] for wname in weather_station_order_in_array]), weather_station_order_in_array
    
