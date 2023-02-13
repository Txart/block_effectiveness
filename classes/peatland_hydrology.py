import numpy as np
import networkx as nx
import rasterio
from rasterio import fill
import fipy as fp
from fipy import numerix
import copy
from tqdm import tqdm
import multiprocessing as mp
import matplotlib.pyplot as plt
import geopandas as gpd
from scipy.spatial import distance
from scipy.interpolate import interp1d

import cwl_utilities
import utilities
from classes.channel_network import ChannelNetwork
from classes.channel_hydrology import CWLHydroParameters
from classes.peatland import Peatland
from classes.peat_hydro_params import PeatlandHydroParameters
from classes.parameterizations import AbstractParameterization

# %%


class AbstractPeatlandHydro:
    def __init__(self,
                 peatland: Peatland,
                 peat_hydro_params: PeatlandHydroParameters,
                 parameterization: AbstractParameterization,
                 channel_network: ChannelNetwork,
                 cwl_params: CWLHydroParameters,
                 zeta_diri_bc=None,
                 force_ponding_storage_equal_one=False) -> None:
        """This class is abstract. Always create one of its subclasses.

        Class for WTD hydrological simulations.

        Args:
            peatland (Peatland): [description]
            peat_hydro_params (PeatlandHydroParameters): [description]
            channel_network (ChannelNetwork): [description]
            cwl_params (CWLHydroParameters): [description]
            zeta_diri_bc (None or float, optional): If None, no flux Neumann BCs are applied. If float, constant
                dirichlet BC is applied everywhere. Default: None.
            force_ponding_storage_equal_one (bool, optional): If True, forces S=1 for water above the surface. Careful:
                if S becomes discontinuous, the hydro model probably won't converge. So, set to True only if S=1 at the surface.  

        """

        self.pl = peatland
        self.ph_params = peat_hydro_params
        self.parameterization = parameterization
        self.cn = channel_network
        self.cn_params = cwl_params
        self.zeta_diri_bc = zeta_diri_bc
        self.force_ponding_storage_equal_one = force_ponding_storage_equal_one

        # Defined by subclasses
        self.dem = None
        self.sourcesink = None
        self.depth = None
        self.mesh = None

        # print stuff
        self.verbose = False

        pass

    def zeta_from_h(self, h):
        return self.parameterization.zeta_from_h(h, self.dem) 

    def h_from_zeta(self, zeta):
        return self.parameterization.h_from_zeta(zeta, self.dem) 

    def create_zeta_from_h(self, h):
        zeta_values = self.zeta_from_h(h).value
        return fp.CellVariable(
            name='zeta', mesh=self.mesh, value=zeta_values, hasOld=True)

    def create_h_from_zeta(self, zeta):
        theta = self.h_from_zeta(zeta)
        return fp.CellVariable(
            name='h (wtd)', mesh=self.mesh, value=theta.value, hasOld=True)

    def _check_initial_wtd_variables(self):
        try:
            _ = self.sourcesink
        except:
            raise ValueError(
                f'Water source (P-ET) not specified. Defaulting to P - ET = {default_source} mm/day')

        return None


    def _has_converged(self, vector1, vector2) -> bool:
        return np.linalg.norm(vector2 - vector1) < np.linalg.norm(vector1)*self.ph_params.rel_tol + self.ph_params.abs_tol


    def _set_equation(self, h):

        # diffussivity mask: diff=0 in faces between canal cells. 1 otherwise.
        mask = 1*(self.canal_mask.arithmeticFaceValue < 0.9)

        storage = self.parameterization.storage(h, self.dem)

        transmissivity = self.parameterization.transmissivity(h, self.dem, self.depth)
        # Apply mask
        # diff_coeff.faceValue is a FaceVariable
        trans_face = transmissivity.faceValue * mask

        eq = (fp.TransientTerm(coeff=storage) ==
              fp.DiffusionTerm(coeff=trans_face) + self.sourcesink
              )

        return eq


    def run(self, h):
        self._check_initial_wtd_variables()

        if self.zeta_diri_bc is not None:
            # Diri BC in h. Compute h from zeta
            h_face_values = self.parameterization.h_from_zeta(
                self.zeta_diri_bc, self.dem.faceValue)
            # constrain theta with those values
            h.constrain(h_face_values, where=self.mesh.exteriorFaces)

        eq = self._set_equation(h)

        residue = 1e10

        for sweep_number in range(self.ph_params.max_sweeps):
            old_residue = copy.deepcopy(residue)
            residue = eq.sweep(var=h, dt=self.ph_params.dt)

            if self.verbose:
                print(sweep_number, residue)

            if residue < self.ph_params.fipy_desired_residual:
                break

        if sweep_number == self.ph_params.max_sweeps - 1:
            raise RuntimeError(
                'WTD computation did not converge in specified amount of sweeps')
        else:
            h.updateOld()

            return h


class GmshMeshHydro(AbstractPeatlandHydro):
    def __init__(self, mesh_fn: str,
                 peatland: Peatland,
                 peat_hydro_params: PeatlandHydroParameters,
                 parameterization: AbstractParameterization,
                 channel_network: ChannelNetwork,
                 cwl_params: CWLHydroParameters,
                 zeta_diri_bc=None) -> None:
        super().__init__(peatland, peat_hydro_params, parameterization,
                         channel_network, cwl_params,
                         zeta_diri_bc=zeta_diri_bc)

        self.mesh = fp.Gmsh2D(mesh_fn)

        mesh_centroids_coords = np.column_stack(self.mesh.cellCenters.value)

        self.mesh_centroids_gdf = self._create_geodataframe_of_mesh_centroids()
        
        self.gdf_mesh_centers_corresponding_to_canals = self._get_geodataframe_mesh_centers_corresponding_to_canals()
        canal_mesh_cell_indices = list(self.gdf_mesh_centers_corresponding_to_canals.index)
        
        # Geometrical quantities for cells and canals
        mean_cell_centroid_distance = self.mesh.scaledCellToCellDistances.mean()
        self.avg_canal_area = mean_cell_centroid_distance * self.cn.B # assuming rectangular shape
        self.avg_cell_area = np.sqrt(3)/4 * mean_cell_centroid_distance**2 # approximating equilateral triangles
        
        # Canal mask
        canal_mask_value = np.zeros(shape=mesh_centroids_coords.shape[0], dtype=int)
        canal_mask_value[canal_mesh_cell_indices] = True
        self.canal_mask = fp.CellVariable(mesh=self.mesh, value=canal_mask_value)
        
        # Maps from mesh to canals and viceversa
        self.mesh_cell_index_to_canal_node_number = self._map_canal_mesh_cells_to_canal_nodes()
        self.canal_node_numbers_to_mesh_indices = self._map_canal_nodes_to_mesh_cells()

        # Link mesh cell centers with canal network nodes
        self.burn_mesh_information_as_canal_graph_attributes()

        dem_value = utilities.sample_raster_from_coords(
            # raster_filename=self.pl.fn_dem,
            raster_filename=self.pl.fn_dem,
            coords=mesh_centroids_coords)
        self.dem = fp.CellVariable(
            name='dem', mesh=self.mesh, value=dem_value, hasOld=False)

        depth_value = utilities.sample_raster_from_coords(
            raster_filename=self.pl.fn_depth, coords=mesh_centroids_coords)
        depth_value[depth_value < 4] = 4  # cutoff depth
        self.depth = fp.CellVariable(
            name='depth', mesh=self.mesh, value=depth_value, hasOld=False)

        # Several weather stations
        if self.ph_params.use_several_weather_stations:
            weather_station_coords_array, self.weather_station_order_in_array = self.pl.read_weather_station_locations()

            self.weather_station_weighting_avg_matrix = self.produce_normalized_weather_station_weighting_average_matrix(
                weather_station_coords_array)

        pass

    def create_initial_zeta_variable_from_raster(self, raster_fn):
        mesh_centroids_coords = np.column_stack(self.mesh.cellCenters.value)
        zeta_values = utilities.sample_raster_from_coords(
            raster_filename=raster_fn,
            coords=mesh_centroids_coords)
        return fp.CellVariable(
            name='zeta', mesh=self.mesh, value=zeta_values, hasOld=True)

    def _create_geodataframe_of_mesh_centroids(self):
        return gpd.GeoDataFrame(geometry=gpd.points_from_xy(self.mesh.x, self.mesh.y))
    
    def _get_geodataframe_mesh_centers_corresponding_to_canals(self):
            gdf_mesh_centers_corresponding_to_canals = gpd.read_file(
                self.pl.fn_canal_mask_mesh_centroids)
        
            canal_mesh_cell_indices = utilities.find_nearest_point_in_other_geodataframe(
                gdf_mesh_centers_corresponding_to_canals['geometry'], self.mesh_centroids_gdf['geometry'])['B'].values
            
            return gdf_mesh_centers_corresponding_to_canals.set_index(canal_mesh_cell_indices)
    
    def _get_nearest_canal_node_for_each_canal_mesh_cell(self):
        # Build gdf from canal network node attributes
        # First, create a dataframe with coords
        graph_df = utilities.convert_networkx_graph_to_dataframe(self.cn.graph)
        canal_nodes_gdf = gpd.GeoDataFrame(
            graph_df, geometry=gpd.points_from_xy(graph_df.x, graph_df.y))
        # Sort according to index. This sorting is very important for finding the mesh match later!
        canal_nodes_gdf = canal_nodes_gdf.sort_index()
        
        return utilities.find_nearest_point_in_other_geodataframe(self.gdf_mesh_centers_corresponding_to_canals['geometry'],
                                                                  canal_nodes_gdf['geometry'])

    def _map_canal_mesh_cells_to_canal_nodes(self):
        return self._get_nearest_canal_node_for_each_canal_mesh_cell()['B'].to_dict()
          
        
    def _get_nearest_mesh_cell_for_each_canal_node(self):
        # Build gdf from canal network node attributes
        # First, create a dataframe with coords
        graph_df = utilities.convert_networkx_graph_to_dataframe(self.cn.graph)
        canal_nodes_gdf = gpd.GeoDataFrame(
            graph_df, geometry=gpd.points_from_xy(graph_df.x, graph_df.y))
        # Sort according to index. This sorting is very important for finding the mesh match later!
        canal_nodes_gdf = canal_nodes_gdf.sort_index()

        return utilities.find_nearest_point_in_other_geodataframe(canal_nodes_gdf['geometry'], self.mesh_centroids_gdf['geometry'])
    
    def _map_canal_nodes_to_mesh_cells(self):
        return self._get_nearest_mesh_cell_for_each_canal_node()['B'].to_dict()
    
    def burn_mesh_information_as_canal_graph_attributes(self):
        # Dict of attributes to add to the canal network nodes
        mesh_node_attributes = {}
        for n in range(self.cn.n_nodes):
            mesh_node_attributes[n] = {'mesh_index': self.canal_node_numbers_to_mesh_indices[n]}
        nx.set_node_attributes(self.cn.graph, mesh_node_attributes)

        return None

        return eq

    def create_fipy_variable_from_graph_attribute(self, graph_attr_name):
        canal_fipy = fp.CellVariable(mesh=self.mesh, value=0.)
        canal_values_in_mesh = canal_fipy.value
        
        for mesh_index, canal_node in self.mesh_cell_index_to_canal_node_number.items():
            canal_values_in_mesh[mesh_index] = self.cn.graph.nodes()[canal_node][graph_attr_name]
            
        canal_fipy.value = canal_values_in_mesh

        return canal_fipy

    def create_fipy_variable_from_channel_network_variable(self, channel_network_var):
        canal_fipy = fp.CellVariable(mesh=self.mesh, value=0.)
        canal_values_in_mesh = canal_fipy.value

        cn_dict = self.cn.from_nparray_to_nodedict(channel_network_var)
        # Step 1) we put nearest canal node value at each mesh cell touched by the channel network.
        for mesh_index, canal_node in self.mesh_cell_index_to_canal_node_number.items():
            canal_values_in_mesh[mesh_index] = cn_dict[canal_node]

        canal_fipy.setValue = canal_values_in_mesh

        return canal_fipy

    def _get_fipy_variable_values_at_graph_nodes(self, fipy_var):
        # This formulation is a bit more cumbersome, but much more efficient
        fipy_var_values_at_canal_nodes = fipy_var.value[list(self.canal_node_numbers_to_mesh_indices.values())]
        return {canal_node: value for canal_node, value in zip(self.canal_node_numbers_to_mesh_indices.keys(), fipy_var_values_at_canal_nodes)}

    def burn_fipy_variable_value_in_graph_attribute(self, fipy_var, graph_attr_name):
        dic_of_values = self._get_fipy_variable_values_at_graph_nodes(fipy_var)
        nx.set_node_attributes(
            self.cn.graph, values=dic_of_values, name=graph_attr_name)
        return None

    def create_uniform_fipy_var(self, uniform_value, var_name):
        return fp.CellVariable(name=var_name, mesh=self.mesh, value=uniform_value)

    def set_canal_values_in_fipy_var(self, channel_network_var, fipy_var):
        cn_fipy_var = self.create_fipy_variable_from_channel_network_variable(
            channel_network_var)
        canal_mask_values = self.canal_mask.value.astype(bool)
        fipy_var.setValue(cn_fipy_var.value, where=canal_mask_values)

        return None

    def convert_h_to_y_at_canals(self, h_fipy_var):
        # h and y are the same 
        h_dict = self._get_fipy_variable_values_at_graph_nodes(
            fipy_var=h_fipy_var)
        y_at_canals = {node: h_value for node, h_value in h_dict.items()}

        return y_at_canals

    def produce_normalized_weather_station_weighting_average_matrix(self, weather_station_coords_array):
        # each row is a different mesh cell center,
        # there are 6 columns, with the distance to each weather station, ordered according to weather_station_order_in_array
        # E.g., distance_matrix[0] = distances of mesh cell center number 0 to all 6 weather stations

        # Weighting average is prop to 1/distance
        weighting_avg_matrix = 1 / \
            (distance.cdist(weather_station_coords_array,
             self.mesh.cellCenters.value.T).T)
        # Normalize
        return (weighting_avg_matrix.T/weighting_avg_matrix.sum(axis=1)).T

    def set_sourcesink_variable(self, value):
        if self.ph_params.use_several_weather_stations:
            self._set_sourcesink_variable_from_several_weather_stations(
                dictionary_of_values=value)
        else:
            self._set_uniform_sourcesink_variable(p_minus_et=value)
        return None

    def _set_sourcesink_variable_from_several_weather_stations(self, dictionary_of_values):
        if len(dictionary_of_values) != self.weather_station_weighting_avg_matrix.shape[1]:
            raise ValueError(
                'The number of P-ET values must equal the number of weather stations')
        p_minus_et_measured_at_stations = np.array(
            [dictionary_of_values[wname] for wname in self.weather_station_order_in_array])
        p_minus_et_values = self.weather_station_weighting_avg_matrix @ p_minus_et_measured_at_stations

        self.sourcesink = fp.CellVariable(
            name='P - ET', mesh=self.mesh, value=p_minus_et_values)
        return None

    def _set_uniform_sourcesink_variable(self, p_minus_et: float):
        self.sourcesink = fp.CellVariable(
            name='P - ET', mesh=self.mesh, value=p_minus_et)
        return None

    def compute_pan_ET_from_ponding_water(self, zeta):
        # pan ET stands for increased ET when WT is above the surface.
        # Here implemented as a linear function of ponding WT height
        MAX_PAN_ET_mm_per_day = 0.004
        ZETA_FOR_MAX_PAN_ET_metres = 0  # zeta at which max pan ET occurs.
        pan_ET_func = interp1d(x=[-0.1, ZETA_FOR_MAX_PAN_ET_metres], y=[0, MAX_PAN_ET_mm_per_day],
                               bounds_error=False, fill_value=(0, MAX_PAN_ET_mm_per_day))

        return pan_ET_func(zeta)


    def _create_geodataframe_from_fipy_variable(self, fipy_var):
        x_cell_centers, y_cell_centers = self.mesh.cellCenters.value
        return gpd.GeoDataFrame({'val': fipy_var},
                                geometry=gpd.points_from_xy(x_cell_centers, y_cell_centers))

        
    def distribute_canal_ponding_water_throughout_cell(self, zeta_canal_nodedict, ponding_canal_zeta_nodedict):
        # input:
        #   - dictionary of (canal node, zeta at node)
        #   - dictionary with the same structure that contain only those canal nodes that have ponding water
        
        for node, zeta in ponding_canal_zeta_nodedict.items():
            zeta_canal_nodedict[node] = self.avg_canal_area[node]/self.avg_cell_area * zeta
            
        return zeta_canal_nodedict
    

    def save_fipy_var_in_raster_file(self, fipy_var, out_raster_fn, interpolation:str):
        template_raster_fn = self.pl.fn_template_output_raster
        gdf = self._create_geodataframe_from_fipy_variable(fipy_var)

        raster = utilities.rasterize_geodataframe_with_template(template_raster_fn, out_raster_fn,
                                                                gdf, gdf_column_name='val',
                                                                interpolation=interpolation)

        # fix interpolation stuff
        interpolation_mask = np.logical_or(
            (self.pl.template_output_raster < 0), (raster > -1e5))
        interpolated = fill.fillnodata(raster, mask=interpolation_mask)
        interpolated[interpolated < -1e10] = interpolated[0, 0]

        raster = interpolated

        utilities.write_raster_from_array(array_to_rasterize=raster,
                                          out_filename=out_raster_fn,
                                          template_raster_filename=template_raster_fn)

        return None

    def imshow(self, fipy_variable):
        plt.figure(figsize=(9, 8), dpi=200)
        viewer = fp.Viewer(vars=fipy_variable,
                           datamin=fipy_variable.value.min(),
                           datamax=fipy_variable.value.max())
        viewer.plot()

        return None


def set_up_peatland_hydrology(mesh_fn,
                              zeta_diri_bc,
                              peatland: Peatland,
                              peat_hydro_params: PeatlandHydroParameters,
                              channel_network: ChannelNetwork,
                              cwl_params: CWLHydroParameters,
                              parameterization: AbstractParameterization):
    """Chooses right subclass to implement, depending on the origin of the mesh

    Args:
        mesh_fn (str, optional): Path to a .msh mesh generated by gmsh. If equal to 'fipy', a nx by ny rectangular grid is used. Defaults to None.
. Defaults to 'fipy'.

    Return:
        [class]: The right subclass of AbstractPeatlandHydro class to implement
    """
    return GmshMeshHydro(mesh_fn=mesh_fn,
                         peatland=peatland,
                         peat_hydro_params=peat_hydro_params,
                         parameterization=parameterization,
                         channel_network=channel_network,
                         cwl_params=cwl_params,
                         zeta_diri_bc=zeta_diri_bc)
