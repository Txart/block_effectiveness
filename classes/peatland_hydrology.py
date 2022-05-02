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

# %%


class PeatlandHydroParameters:
    def __init__(self, dt, dx, nx, ny,
                 max_sweeps, fipy_desired_residual,
                 s1, s2, t1, t2,
                 average_mesh_triangle_side_length,
                 use_several_weather_stations) -> None:
        self.dt = dt
        self.dx = dx
        self.nx = nx
        self.ny = ny
        self.max_sweeps = max_sweeps
        self.fipy_desired_residual = fipy_desired_residual

        # parameters for S and T functions. Also, theta to zeta
        self.s1 = s1
        self.s2 = s2
        self.t1 = t1
        self.t2 = t2
        
        # Compute P-ET based on weighted average distance of several weather stations
        self.use_several_weather_stations = use_several_weather_stations
        
        # Triangular mesh element size for computing mesh cells that are touched by channel network
        self.average_mesh_triangle_side_length = average_mesh_triangle_side_length

        pass


class AbstractPeatlandHydro:
    def __init__(self,
                 peatland: Peatland,
                 peat_hydro_params: PeatlandHydroParameters,
                 channel_network: ChannelNetwork,
                 cwl_params: CWLHydroParameters,
                 model_coupling='darcy',
                 use_scaled_pde='False',
                 zeta_diri_bc=None,
                 force_ponding_storage_equal_one=False) -> None:
        """This class is abstract. Always create one of its subclasses.

        Class for WTD hydrological simulations.

        Args:
            peatland (Peatland): [description]
            peat_hydro_params (PeatlandHydroParameters): [description]
            channel_network (ChannelNetwork): [description]
            cwl_params (CWLHydroParameters): [description]
            model_coupling (str, optional): Can either be 'darcy' or 'dirichlet'. If 'darcy' the canals
                are not used as boundary conditions, instead, boussinesq eq, i.e., Darcy's Law, is applied on them as well.
                If 'dirichlet' canals are used as Dirichlet fixed head boundary conditions in the 
                groundwater simulation step. Defaults to 'darcy'.
            use_scaled_pde (bool, optional): Run fipy hydrology with the scaled version of the pde. Might help in
                some cases when numbers become too large. Default: False.
            zeta_diri_bc (None or float, optional): If None, no flux Neumann BCs are applied. If float, constant
                dirichlet BC is applied everywhere. Default: None.
            force_ponding_storage_equal_one (bool, optional): If True, forces S=1 for water above the surface. Careful:
                if S becomes discontinuous, the hydro model probably won't converge. So, set to True only if S=1 at the surface.  

        """

        self.pl = peatland
        self.ph_params = peat_hydro_params
        self.cn = channel_network
        self.cn_params = cwl_params
        self.model_coupling = model_coupling
        self.use_scaled_pde = use_scaled_pde
        self.zeta_diri_bc = zeta_diri_bc
        self.force_ponding_storage_equal_one = force_ponding_storage_equal_one

        self._check_model_coupling_choice()

        # Defined by subclasses
        self.sourcesink = None
        self.depth = None
        
        # print stuff
        self.verbose = False

        pass

    def _check_model_coupling_choice(self):
        if self.model_coupling == 'dirichlet':
            raise Warning(
                'Feedback from wtd to next iteration cwl not yet implemented in dirichlet coupling mode.')
        utilities.is_value_of_variable_possible(
            self.model_coupling, possible_values=['dirichlet', 'darcy'])

    # Old stuff
    # def h_from_theta(self, theta):
    #     return numerix.sqrt(2*theta/self.ph_params.s2)

    # def theta_from_h(self, h):
    #     return self.ph_params.s2/2 * h**2

    # New stuff
    def h_from_theta(self, theta):
        return numerix.log(self.ph_params.s2/self.ph_params.s1*numerix.exp(self.ph_params.s2*self.dem)*theta + 1)/self.ph_params.s2

    def theta_from_h(self, h):
        return self.ph_params.s1/self.ph_params.s2*(numerix.exp(self.ph_params.s2*(h-self.dem)) - numerix.exp(-self.ph_params.s2*self.dem)) 
    
    def zeta_from_h(self, h):
        return h - self.dem

    def h_from_zeta(self, zeta):
        return zeta + self.dem
    
    def theta_from_zeta(self, zeta):
        return self.ph_params.s1/self.ph_params.s2*(numerix.exp(self.ph_params.s2*zeta) - numerix.exp(-self.ph_params.s2*self.dem))

    def zeta_from_theta(self, theta):
        return numerix.log(self.ph_params.s2*theta/self.ph_params.s1 + numerix.exp(-self.ph_params.s2*self.dem))/self.ph_params.s2

    def create_zeta_from_theta(self, theta):
        zeta_values = self.zeta_from_theta(theta).value
        return fp.CellVariable(
            name='zeta', mesh=self.mesh, value=zeta_values, hasOld=True)

    def create_theta_from_zeta(self, zeta):
        theta = self.theta_from_zeta(zeta)
        theta = fp.CellVariable(
            name='theta', mesh=self.mesh, value=theta.value, hasOld=True)
        return theta
    
    def _check_initial_wtd_variables(self):
        try:
            _ = self.sourcesink
        except:
            raise ValueError(
                f'Water source (P-ET) not specified. Defaulting to P - ET = {default_source} mm/day')

        return None

    def _has_converged(self, vector1, vector2) -> bool:
        return np.linalg.norm(vector2 - vector1) < np.linalg.norm(vector1)*self.ph_params.rel_tol + self.ph_params.abs_tol

    def _diffusivity_cell_variable(self, theta):
        diffusivity = self.dif(theta.value, self.depth.value)
        return fp.CellVariable(mesh=self.mesh, value=diffusivity)

    def _diffusivity_zero_between_canal_pixels(self, theta):
        canal_faces_mask = fp.FaceVariable(
            mesh=self.mesh, value=self.canal_mask.arithmeticFaceValue.value)

        diff_cell = self._diffusivity_cell_variable(theta)
        # Value of the diffusion faceVariable with arithmetic face value.
        diff_face_value = fp.FaceVariable(
            mesh=self.mesh, value=diff_cell.arithmeticFaceValue.value).value
        # Constrain diffusion between adjacent canal pixels to be zero using the mask.
        # Logic: Arithmetic mean in the face between two adjacent canal pixels
        # whose diffusivity = 0, is equal to (0+0)/2 = 0, while all the rest of faces
        # between pixels are different from 0.
        #diff_face_value[self.canal_mask_face.value == 0] = 0.0
        diff_face_value = diff_face_value * canal_faces_mask.value

        return fp.FaceVariable(mesh=self.mesh, value=diff_face_value)

    def _set_equation(self, theta):
        if self.model_coupling == 'darcy':
            eq = self._set_equation_darcymode(theta)
        elif self.model_coupling == 'dirichlet':
            eq = self._set_equation_dirimode(theta)
        else:
            raise ValueError('coupling mode not understood. Aborting.')

        return eq

    def _run_theta(self, theta):
        if self.zeta_diri_bc is not None:
            # Diri BC in theta. Compute theta from zeta
            theta_face_values = self.ph_params.s1/self.ph_params.s2*(numerix.exp(self.ph_params.s2*self.zeta_diri_bc) - numerix.exp(-self.ph_params.s2*self.dem.faceValue)) 
            # constrain theta with those values
            theta.constrain(theta_face_values, where=self.mesh.exteriorFaces)
        
        eq = self._set_equation(theta)

        residue = 1e10
        sweep_number = 0
        while residue > self.ph_params.fipy_desired_residual:
            residue = eq.sweep(var=theta, dt=self.ph_params.dt)
            sweep_number += 1
            
            if self.verbose:
                print(sweep_number, residue)

            if sweep_number > self.ph_params.max_sweeps:
                raise RuntimeError(
                    'WTD computation did not converge in specified amount of sweeps')

        theta.updateOld()

        return theta
    
    def _run_theta_scaled(self, theta):
        
        s1 = self.ph_params.s1
        s2 = self.ph_params.s2
        t1 = self.ph_params.t1
        t2 = self.ph_params.t2
        diff_coeff = (
                t1/s1**(t2/s2)*(s1*numerix.exp(-self.dem*s2) + s2*theta)**(t2/s2-1) -
                t1*numerix.exp(-self.b*t2)/(s1*numerix.exp(-self.dem*s2) + s2*theta)
                )
        
        # SCALES
        theta_C = theta.value.max()
        dif_C = diff_coeff.value.max()
        source_C = theta_C * dif_C
        t_C = 1/dif_C
        
        # Apply scales
        dt_scaled = self.ph_params.dt / t_C
        
        # SCALE INI COND
        theta_scaled = fp.CellVariable(
            name='scaled theta', mesh=self.mesh, value=theta.value / theta_C, hasOld=True)
        
        if self.zeta_diri_bc is not None:
            # Boundary conditions scaled as well!
            theta_face_values = self.ph_params.s1/self.ph_params.s2*(numerix.exp(self.ph_params.s2*self.zeta_diri_bc) - numerix.exp(-self.ph_params.s2*self.dem.faceValue)) 
            # constrain theta with those values
            theta_scaled.constrain(theta_face_values/theta_C, where=self.mesh.exteriorFaces)      

        # The equation creation funciton needs original, non-scaled theta
        eq = self._set_equation_darcymode_scaled(theta)

        residue = 1e10
        sweep_number = 0
        while residue > self.ph_params.fipy_desired_residual:
            residue = eq.sweep(var=theta_scaled, dt=dt_scaled)
            sweep_number += 1
            
            if self.verbose:
                print(sweep_number, residue)

            if sweep_number > self.ph_params.max_sweeps:
                raise RuntimeError(
                    'WTD computation did not converge in specified amount of sweeps')

        theta_scaled.updateOld()
        
        # SCALE BACK
        theta = fp.CellVariable(
            name='theta', mesh=self.mesh, value=theta_scaled.value * theta_C, hasOld=True)

        return theta

    def _run_theta_halt_with_var_relative_difference(self, fipy_var):
        # Sometimes, e.g. when there are internal BC,
        # the residual calculated by fipy is meaningless and we have to
        # use another stopping criterion for convergence.
        # Here, I use the relative difference between 2 timesteps of the solution variable

        eq = self._set_equation(fipy_var)
        for sweep_number in range(self.ph_params.max_sweeps):
            theta_previous = copy.copy(fipy_var.value)
            eq.sweep(var=fipy_var, dt=self.ph_params.dt)

            print('theta diff = ', np.linalg.norm(
                theta_previous - fipy_var.value))
            # Stopping criterion
            if np.linalg.norm(theta_previous - fipy_var.value) < 1e-7:
                return fipy_var

        raise RuntimeError(
            'WTD computation did not converge in specified amount of sweeps')

        return None

    def run(self, fipy_var, mode='theta'):
        self._check_initial_wtd_variables()

        if mode == 'theta':
            if self.model_coupling == 'darcy':
                if self.use_scaled_pde:
                    return self._run_theta_scaled(theta=fipy_var)
                else:
                    return self._run_theta(theta=fipy_var)
            if self.model_coupling == 'dirichlet':
                return self._run_theta_halt_with_var_relative_difference(fipy_var)
        elif mode == 'standard':
            raise NotImplementedError('standard hydro not implemenmted yet.')
        else:
            raise ValueError('mode has to be either "theta" or "standard"')

    def _set_equation_darcymode(self, theta):
        # Implemented by subclasses
        return None

    def _set_equation_dirimode(self, theta):
        # Implemented by subclasses
        return None
    
    def create_uniform_fipy_var(self):
        # Implemented by subclasses
        return None

    def set_sourcesink_variable(self, p_minus_et: float):
        # Implemented by subclasses
        return None


class GmshMeshHydro(AbstractPeatlandHydro):
    def __init__(self, mesh_fn: str,
                 peatland: Peatland,
                 peat_hydro_params: PeatlandHydroParameters,
                 channel_network: ChannelNetwork,
                 cwl_params: CWLHydroParameters,
                 model_coupling='darcy',
                 use_scaled_pde=False,
                 zeta_diri_bc=None,
                 force_ponding_storage_equal_one=False) -> None:
        super().__init__(peatland, peat_hydro_params, channel_network,
                         cwl_params, model_coupling=model_coupling, use_scaled_pde=use_scaled_pde,
                         zeta_diri_bc=zeta_diri_bc, force_ponding_storage_equal_one=force_ponding_storage_equal_one)

        self.mesh = fp.Gmsh2D(mesh_fn)

        mesh_centroids_coords = np.column_stack(self.mesh.cellCenters.value)
        self.mesh_centroids_gdf = self._create_geodataframe_of_mesh_centroids()
        nearest_mesh_cell_for_each_canal_point = self._get_nearest_mesh_cell_for_each_canal_point()
        self.canal_node_numbers_to_mesh_indices = nearest_mesh_cell_for_each_canal_point['B'].to_dict()
        self.mesh_cell_index_to_canal_node_number = self._get_nearest_canal_node_for_each_mesh_cell()['B'].to_dict()

        self.mesh_indices_touched_by_channel_network = list(self._get_mesh_cells_touched_by_canal_network()['point'])

        # Canal mask
        canal_mask_value = np.zeros(
            shape=mesh_centroids_coords.shape[0], dtype=int)
        canal_mask_value[self.mesh_indices_touched_by_channel_network] = True
        self.canal_mask = fp.CellVariable(
            mesh=self.mesh, value=canal_mask_value)

        # Each mesh cell can contain more than one canal node.
        # In those cases, the arithmetic mean will be computed. For that, we
        # first count the amount of points in a single mesh, and then multiply
        # the cwl by 1/amount
        self.mesh_index_count = nearest_mesh_cell_for_each_canal_point['B'].value_counts().to_dict()

        # Link mesh cell centers with canal network nodes
        self.burn_mesh_information_as_canal_graph_attributes()

        # Sample dem ad depth rasters at mesh centroids from corresponding rasters
        if self.ph_params.bigger_dem_raster_fn is not None:
            dem_raster_fn_to_sample_mesh = self.ph_params.bigger_dem_raster_fn
        else:
            dem_raster_fn_to_sample_mesh = self.pl.fn_dem
        dem_value = utilities.sample_raster_from_coords(
            # raster_filename=self.pl.fn_dem,
            raster_filename=dem_raster_fn_to_sample_mesh,
            coords=mesh_centroids_coords)
        self.dem = fp.CellVariable(
            name='dem', mesh=self.mesh, value=dem_value, hasOld=False)
        
        depth_value = utilities.sample_raster_from_coords(raster_filename=self.pl.fn_depth, coords=mesh_centroids_coords)
        depth_value[depth_value < 4] = 4 # cutoff depth
        self.b = fp.CellVariable(name='depth', mesh=self.mesh, value=depth_value, hasOld=False)
        
        # Several weather stations
        if self.ph_params.use_several_weather_stations:
            weather_station_coords_array, self.weather_station_order_in_array = self.pl.read_weather_station_locations()
        
            self.weather_station_weighting_avg_matrix = self.produce_normalized_weather_station_weighting_average_matrix(weather_station_coords_array)

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

    def _get_nearest_mesh_cell_for_each_canal_point(self):
        # Build gdf from canal network node attributes
        # First, create a dataframe with coords
        graph_df = utilities.convert_networkx_graph_to_dataframe(self.cn.graph)
        canal_nodes_gdf = gpd.GeoDataFrame(
            graph_df, geometry=gpd.points_from_xy(graph_df.x, graph_df.y))
        # Sort according to index. This sorting is very important for finding the mesh match later!
        canal_nodes_gdf = canal_nodes_gdf.sort_index()
        # When giving a mesh from file, map between canal nodes and the mesh has to be computed
        mesh_centroids_gdf = self._create_geodataframe_of_mesh_centroids()
        # copy index as a column in order to store it after the nearest neighbour search
        mesh_centroids_gdf['mesh_index'] = mesh_centroids_gdf.index

        return utilities.find_nearest_point_in_other_geodataframe(canal_nodes_gdf['geometry'], mesh_centroids_gdf)

    def _get_nearest_canal_node_for_each_mesh_cell(self):
        graph_df = utilities.convert_networkx_graph_to_dataframe(self.cn.graph)
        canal_nodes_gdf = gpd.GeoDataFrame(
            graph_df, geometry=gpd.points_from_xy(graph_df.x, graph_df.y))
        # Sort according to index. This sorting is very important for finding the mesh match later!
        canal_nodes_gdf = canal_nodes_gdf.sort_index()
        
        return utilities.find_nearest_point_in_other_geodataframe(self.mesh_centroids_gdf['geometry'], canal_nodes_gdf['geometry'])
        
    
    def _get_mesh_cells_touched_by_canal_network(self):
        cn_lines = gpd.read_file(self.pl.fn_channel_network_lines)
        try: # if multilinestring, convert to linestring. If linestring, throws exception and do nothing
            cn_lines_single = utilities.multilinestring_to_linestring(cn_lines)['geometry']
        except:
            cn_lines_single = cn_lines
            
        dist_df = utilities.find_distance_of_points_in_geodataframe_to_nearest_line_in_geodataframe(
                                                gdf_points=self.mesh_centroids_gdf, gdf_lines=cn_lines_single)
    
        threshold_size = self.ph_params.average_mesh_triangle_side_length * np.sqrt(3)/4 # based on geometry of equilateral triangles. Distance from triangle center to edge.
        dist_df = dist_df[dist_df['distance'] < threshold_size]
        
        return dist_df


    def burn_mesh_information_as_canal_graph_attributes(self):
        # Dict of attributes to add to the canal network nodes
        mesh_node_attributes = {}
        for n in range(self.cn.n_nodes):
            mesh_node_attributes[n] = {'mesh_index': self.canal_node_numbers_to_mesh_indices[n],
                                       'mesh_averaging_weight': 1/self.mesh_index_count[self.canal_node_numbers_to_mesh_indices[n]]}

        nx.set_node_attributes(self.cn.graph, mesh_node_attributes)

        return None

    def _set_equation_darcymode(self, theta):
        # diffussivity mask: diff=0 in faces between canal cells. 1 otherwise.
        mask = 1*(self.canal_mask.arithmeticFaceValue < 0.9)
        
        # New diffusion coefficient, rearranging terms to minimize risk of going beyond machine precision
        s1 = self.ph_params.s1
        s2 = self.ph_params.s2
        t1 = self.ph_params.t1
        t2 = self.ph_params.t2
        diff_coeff = (
                t1/s1**(t2/s2)*(s1*numerix.exp(-self.dem*s2) + s2*theta)**(t2/s2-1) -
                t1*numerix.exp(-self.b*t2)/(s1*numerix.exp(-self.dem*s2) + s2*theta)
                )
        # Set diff coeff with S=1 when water is ponding
        if self.force_ponding_storage_equal_one:
            zeta = self.zeta_from_theta(theta)
            ponding_mask = 1*(zeta > 0) # 1 where ponding, 0 otherwise
            not_ponding_mask = 1 - ponding_mask # 1 where not ponding, 0 otherwise
            
            diff_coeff_ponding = t1*((s2/s1*theta + numerix.exp(-s2*self.dem))**(t2/s2) -
                                     numerix.exp(-t2*self.b)) # when S=1, D=T
            
            diff_coeff = diff_coeff * not_ponding_mask + diff_coeff_ponding * ponding_mask
            
        # Apply mask
        # diff_coeff.faceValue is a FaceVariable
        diff_coeff_face = diff_coeff.faceValue * mask

        largeValue = 1e60
        eq = (fp.TransientTerm() ==
              fp.DiffusionTerm(coeff=diff_coeff_face) + self.sourcesink
              # - fp.ImplicitSourceTerm(self.catch_mask_not * largeValue) +
              #self.catch_mask_not * largeValue * 0.
              )
        return eq
    
    def _set_equation_darcymode_scaled(self, theta):        
       # diffussivity mask: diff=0 in faces between canal cells. 1 otherwise.
        mask = 1*(self.canal_mask.arithmeticFaceValue < 0.9)
        
        # New diffusion coefficient, rearranging terms to minimize risk of going beyond machine precision
        s1 = self.ph_params.s1
        s2 = self.ph_params.s2
        t1 = self.ph_params.t1
        t2 = self.ph_params.t2
        diff_coeff = (
                t1/s1**(t2/s2)*(s1*numerix.exp(-self.dem*s2) + s2*theta)**(t2/s2-1) -
                t1*numerix.exp(-self.b*t2)/(s1*numerix.exp(-self.dem*s2) + s2*theta)
                )
        
        # SCALES
        theta_C = theta.value.max()
        dif_C = diff_coeff.value.max()
        source_C = theta_C * dif_C
        t_C = 1/dif_C
        # x_C = 1 is imposed by gmsh mesh
        
        # apply scales
        source_scaled = self.sourcesink / source_C
        diff_coeff_scaled = diff_coeff / dif_C

        # Apply mask
        # diff_coeff.faceValue is a FaceVariable
        diff_coeff_face_scaled = diff_coeff_scaled.faceValue * mask

        largeValue = 1e60
        eq = (fp.TransientTerm() ==
              fp.DiffusionTerm(coeff=diff_coeff_face_scaled) + source_scaled
              )
        
        return eq

    def _set_equation_dirimode(self, fipy_var):
        # raise NotImplementedError()

        # compute diffusivity. Cell Variable
        diff_coeff = (self.ph_params.t1*numerix.exp(-self.ph_params.t2*self.dem)/numerix.sqrt(2*self.ph_params.s2 * theta) *
                      (numerix.exp(self.ph_params.t2*numerix.sqrt(2*theta/self.ph_params.s2)) - 1))

        largeValue = 1e60
        eq = (fp.TransientTerm() ==
              # fp.DiffusionTerm(coeff=1.)
              fp.DiffusionTerm(coeff=diff_coeff) + self.sourcesink
              - fp.ImplicitSourceTerm(self.canal_mask * largeValue) +
              self.canal_mask * largeValue * fipy_var.value
              )
        return eq

    def create_fipy_variable_from_graph_attribute(self, graph_attr_name):
        canal_fipy = fp.CellVariable(mesh=self.mesh, value=0.)
        canal_values_in_mesh = canal_fipy.value
        for canal_node, mesh_pos in self.canal_node_numbers_to_mesh_indices.items():
            # when more than one canal node in the same mesh cell, arithmetic mean
            # is implemented using the mesh_averaging_weight node attribute
            canal_values_in_mesh[mesh_pos] += (self.cn.graph.nodes()[canal_node][graph_attr_name] *
                                               self.cn.graph.nodes()[canal_node]['mesh_averaging_weight'])
        canal_fipy.value = canal_values_in_mesh

        return canal_fipy
    
    def create_fipy_variable_from_channel_network_variable(self, channel_network_var):
        canal_fipy = fp.CellVariable(mesh=self.mesh, value=0.)
        canal_values_in_mesh = canal_fipy.value
        
        # when more than one canal node in the same mesh cell, arithmetic mean
        # is implemented using the mesh_averaging_weight node attribute
        mesh_avg_weight_vector = self.cn.from_graph_attribute_to_nparray('mesh_averaging_weight')
        mesh_averaged_channel_network_var = mesh_avg_weight_vector * channel_network_var
        mesh_avg_var_nodedict = self.cn.from_nparray_to_nodedict(mesh_averaged_channel_network_var)
        
        for canal_node, mesh_pos in self.canal_node_numbers_to_mesh_indices.items():
            canal_values_in_mesh[mesh_pos] += mesh_avg_var_nodedict[canal_node]
            
        canal_fipy.setValue = canal_values_in_mesh

        return canal_fipy

    def _get_fipy_variable_values_at_graph_nodes(self, fipy_var):
        fipy_var_values_at_canal_nodes = fipy_var.value[list(
            self.canal_node_numbers_to_mesh_indices.values())]
        return {canal_node: value for canal_node, value in zip(self.canal_node_numbers_to_mesh_indices.keys(), fipy_var_values_at_canal_nodes)}

    def burn_fipy_variable_value_in_graph_attribute(self, fipy_var, graph_attr_name):
        dic_of_values = self._get_fipy_variable_values_at_graph_nodes(fipy_var)
        nx.set_node_attributes(
            self.cn.graph, values=dic_of_values, name=graph_attr_name)
        return None

    def create_uniform_fipy_var(self, uniform_value, var_name):
        return fp.CellVariable(name=var_name, mesh=self.mesh, value=uniform_value)

    # def _set_canal_mask_from_cwl_fipy_var(self, cwl_fipy_var):
    #     canal_mask_values = (~np.isclose(cwl_fipy_var.value, 0)).astype(dtype=int)
    #     self.canal_mask = fp.CellVariable(mesh=self.mesh, value=canal_mask_values)
    #     return None

    def set_canal_values_in_fipy_var(self, channel_network_var, fipy_var):
        cn_fipy_var = self.create_fipy_variable_from_channel_network_variable(channel_network_var)
        canal_mask_values = self.canal_mask.value.astype(bool)
        fipy_var.setValue(cn_fipy_var.value, where=canal_mask_values)

        return None

    def convert_zeta_to_y_at_canals(self, zeta_fipy_var):
        # Compute y = zeta + dem
        zeta_dict = self._get_fipy_variable_values_at_graph_nodes(
            fipy_var=zeta_fipy_var)
        dem_dict = nx.get_node_attributes(self.cn.graph, 'DEM')
        y_at_canals = {node: zeta_value +
                       dem_dict[node] for node, zeta_value in zeta_dict.items()}
        return y_at_canals

    def produce_normalized_weather_station_weighting_average_matrix(self, weather_station_coords_array):
        # each row is a different mesh cell center, 
        # there are 6 columns, with the distance to each weather station, ordered according to weather_station_order_in_array
        # E.g., distance_matrix[0] = distances of mesh cell center number 0 to all 6 weather stations
        
        # Weighting average is prop to 1/distance
        weighting_avg_matrix = 1/(distance.cdist(weather_station_coords_array, self.mesh.cellCenters.value.T).T)
        return (weighting_avg_matrix.T/weighting_avg_matrix.sum(axis=1)).T # Normalize
    
    def set_sourcesink_variable(self, value):
        if self.ph_params.use_several_weather_stations:
            self._set_sourcesink_variable_from_several_weather_stations(dictionary_of_values=value)
        else:
            self._set_uniform_sourcesink_variable(p_minus_et=value)
        return None
            
    
    def _set_sourcesink_variable_from_several_weather_stations(self, dictionary_of_values):
        if len(dictionary_of_values) != self.weather_station_weighting_avg_matrix.shape[1]:
            raise ValueError('The number of P-ET values must equal the number of weather stations')
        p_minus_et_measured_at_stations = np.array([dictionary_of_values[wname] for wname in self.weather_station_order_in_array])
        p_minus_et_values = self.weather_station_weighting_avg_matrix @ p_minus_et_measured_at_stations
        
        self.sourcesink = fp.CellVariable(name='P - ET', mesh=self.mesh, value=p_minus_et_values)
        return None
    
    def _set_uniform_sourcesink_variable(self, p_minus_et: float):
        self.sourcesink = fp.CellVariable(name='P - ET', mesh=self.mesh, value=p_minus_et)
        return None
    
    def compute_pan_ET_from_ponding_water(self, zeta):
        # pan ET stands for increased ET when WT is above the surface.
        # Here implemented as a linear function of ponding WT height
        MAX_PAN_ET_mm_per_day = 0.003
        ZETA_FOR_MAX_PAN_ET_metres = 0.3 # zeta at which max pan ET occurs.
        pan_ET_func = interp1d(x=[0, ZETA_FOR_MAX_PAN_ET_metres], y=[0, MAX_PAN_ET_mm_per_day],
                               bounds_error=False, fill_value=(0, MAX_PAN_ET_mm_per_day))
        
        return pan_ET_func(zeta)
    
    
    def set_theta_diriBC_from_zeta_value(self, zeta_diri_BC, theta_fipy):
        dem_face_values = self.dem.faceValue
        # Compute theta from zeta
        theta_face_values = self.ph_params.s1/self.ph_params.s2*(numerix.exp(self.ph_params.s2*zeta_diri_BC) - numerix.exp(-self.ph_params.s2*dem_face_values)) 
        # constrain theta with those values
        theta_fipy.constrain(theta_face_values, self.mesh.exteriorFaces)
        
        return theta_fipy

    def _create_geodataframe_from_fipy_variable(self, fipy_var):
        x_cell_centers, y_cell_centers = self.mesh.cellCenters.value
        return gpd.GeoDataFrame({'val': fipy_var},
                                geometry=gpd.points_from_xy(x_cell_centers, y_cell_centers))

    def save_fipy_var_in_raster_file(self, fipy_var, out_raster_fn, interpolate: bool):
        template_raster_fn = self.pl.fn_dem
        gdf = self._create_geodataframe_from_fipy_variable(fipy_var)

        raster = utilities.rasterize_geodataframe_with_template(template_raster_fn, out_raster_fn,
                                                                gdf, gdf_column_name='val')

        if interpolate:
            interpolation_mask = np.logical_or(
                (self.pl.dem < 0), (raster > -1e5))
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


class RectangularMeshHydro(AbstractPeatlandHydro):

    def __init__(self, peatland: Peatland,
                 peat_hydro_params: PeatlandHydroParameters,
                 channel_network: ChannelNetwork,
                 cwl_params: CWLHydroParameters,
                 model_coupling='darcy') -> None:
        super().__init__(peatland, peat_hydro_params, channel_network,
                         cwl_params, model_coupling=model_coupling)
        self.mesh = fp.Grid2D(dx=self.ph_params.dx, dy=self.ph_params.dx,
                              nx=self.ph_params.nx, ny=self.ph_params.ny)

        dem_value = self.pl.dem.flatten()
        # Avoid problems with hydro variables out of bounds
        dem_value[dem_value < 0] = 5.0
        self.dem = fp.CellVariable(
            name='dem', mesh=self.mesh, value=dem_value, hasOld=False)

        depth_value = self.pl.peat_depth.flatten()
        # Avoid problems with hydro variables out of bounds
        depth_value[depth_value < 1] = 5.0
        self.depth = fp.CellVariable(
            name='peat_depth', mesh=self.mesh, value=depth_value, hasOld=False)

        self.canal_mask = fp.CellVariable(
            mesh=self.mesh, value=self.pl.canal_mask.astype(dtype=int).flatten())
        self.catchment_mask = fp.CellVariable(
            mesh=self.mesh, value=np.array(self.pl.catchment_mask, dtype=int).flatten())
        self.catch_mask_not = fp.CellVariable(
            mesh=self.mesh, value=np.array(~self.pl.catchment_mask, dtype=int).flatten())
        self.catch_mask_not_face = fp.FaceVariable(
            mesh=self.mesh, value=self.catch_mask_not.arithmeticFaceValue.value)
        
        if self.ph_params.use_several_weather_stations:
            raise NotImplementedError("""several weather stations not implemented in rectangular meshes.
                                      This was actually implemented in the 1st project with Forest Carbon.
                                      Look there for answers on how to implement it.
                                      Otherwise, use constant P-ET.""")

        pass

    def set_sourcesink_variable(self, p_minus_et: float):
        sourcearray = p_minus_et * self.catchment_mask.value
        self.sourcesink = fp.CellVariable(
            name='P - ET', mesh=self.mesh, value=sourcearray)
        return None

    def set_internal_canal_values_in_zeta(self):
        # Read y from channel network computation and put it into zeta
        y_in_canals = self.pl.rasterize_graph_attribute_to_dem_shape(
            graph=self.cn.graph, graph_attr='zeta')
        zeta_value = self._reshape_fipy_var_to_array(fipy_var=self.zeta)
        zeta_value[self.pl.canal_mask] = y_in_canals[self.pl.canal_mask]
        self.zeta.value = zeta_value.flatten()

        return None

    def create_uniform_fipy_var(self, uniform_value, var_name):
        value = uniform_value * self.catchment_mask.value
        return fp.CellVariable( name=var_name, mesh=self.mesh, value=zeta_value)


    def _set_equation_darcymode(self, theta):
        # diffussivity mask has 2 parts:
        # Part 1) diff=0 outside catchment. Since harmonicFaceValue, it is also zero in the boundary faces.
        # Part 2) diff=0 in faces between canal cells. 1 otherwise
        mask_part1 = self.catchment_mask.harmonicFaceValue.value
        mask_part2 = 1*(self.canal_mask.arithmeticFaceValue.value > 1e-7)
        mask = mask_part1 * mask_part2

        # compute diffusivity. Cell Variable
        diff_coeff = (self.ph_params.t1*numerix.exp(-self.ph_params.t2*self.depth)/numerix.sqrt(2*self.ph_params.s2) *
                      (numerix.exp(self.ph_params.t2*numerix.sqrt(2*theta/self.ph_params.s2)) - 1) / numerix.sqrt(theta))

        # Apply mask
        # diff_coeff.faceValue is a FaceVariable
        diff_coeff_face = diff_coeff.faceValue * mask

        largeValue = 1e60
        eq = (fp.TransientTerm() ==
              fp.DiffusionTerm(coeff=diff_coeff_face) + self.sourcesink
              #   - fp.ImplicitSourceTerm(self.catch_mask_not * largeValue) +
              #   self.catch_mask_not * largeValue * 0.
              )
        return eq

    def _set_equation_dirimode(self, theta):
        # compute diffusivity. Cell Variable
        diff_coeff = (self.ph_params.t1*numerix.exp(-self.ph_params.t2*self.depth)/numerix.sqrt(2*self.ph_params.s2) *
                      (numerix.exp(self.ph_params.t2*numerix.sqrt(2*theta/self.ph_params.s2)) - 1) / numerix.sqrt(theta))

        largeValue = 1e60
        eq = (fp.TransientTerm() ==
              # fp.DiffusionTerm(coeff=1.)
              fp.DiffusionTerm(coeff=diff_coeff) + self.sourcesink
              - fp.ImplicitSourceTerm(self.canal_mask * largeValue) +
              self.canal_mask * largeValue * theta.value
              - fp.ImplicitSourceTerm(self.catch_mask_not * largeValue) + \
              self.catch_mask_not * largeValue * 0.

              # + fp.DiffusionTerm(coeff=largeValue * catch_mask_not_face)
              # - fp.ImplicitSourceTerm((catch_mask_not_face*largeValue*neumann_bc*mesh.faceNormals).divergence)
              )
        return eq

    def _reshape_fipy_var_to_array(self, fipy_var) -> np.ndarray:
        return fipy_var.value.reshape(
            self.ph_params.nx, self.ph_params.ny)

    def burn_fipy_variable_value_in_graph_attribute(self, fipy_var, graph_attr_name):
        fipy_var_value = self._reshape_fipy_var_to_array(fipy_var=fipy_var)
        self.pl.set_raster_value_to_graph_attribute(
            raster_value=fipy_var_value, graph=self.cn.graph, graph_attr=graph_attr_name)
        return None

    def convert_zeta_to_y_at_canals(self):
        # Compute y = zeta + dem
        zeta_dict = nx.get_node_attributes(self.cn.graph, 'zeta')
        dem_dict = nx.get_node_attributes(self.cn.graph, 'DEM')
        y_at_canals = {node: zeta_value +
                       dem_dict[node] for node, zeta_value in zeta_dict.items()}
        return y_at_canals

    def save_fipy_var_in_raster_file(self, fipy_var, out_raster_fn, interpolate=False):
        if interpolate == True:
            raise Warning(
                'Nothing to interpolate in a 2D rectabgular mesh. \nContinuing without interpolation')
        template_raster_fn = self.pl.fn_dem
        array_to_rasterize = fipy_var.value.reshape(
            self.ph_params.nx, self.ph_params.ny)
        utilities.write_raster_from_array(array_to_rasterize=array_to_rasterize,
                                          out_filename=out_raster_fn,
                                          template_raster_filename=template_raster_fn)

        return None

    def imshow(self, variable):
        plt.figure(figsize=(18, 16), dpi=200)
        plt.imshow(variable.value.reshape(
            self.ph_params.nx, self.ph_params.ny))
        plt.colorbar()

        return None


def set_up_peatland_hydrology(mesh_fn,
                              model_coupling,
                              use_scaled_pde,
                              zeta_diri_bc,
                              force_ponding_storage_equal_one,
                              peatland: Peatland,
                              peat_hydro_params: PeatlandHydroParameters,
                              channel_network: ChannelNetwork,
                              cwl_params: CWLHydroParameters):
    """Chooses right subclass to implement, depending on the origin of the mesh

    Args:
        mesh_fn (str, optional): Path to a .msh mesh generated by gmsh. If equal to 'fipy', a nx by ny rectangular grid is used. Defaults to None.
. Defaults to 'fipy'.

    Return:
        [class]: The right subclass of AbstractPeatlandHydro class to implement
    """
    if mesh_fn == 'fipy':
        if use_scaled_pde:
            raise NotImplementedError('Scaled PDE version is only supported with gmsh mesh')
        return RectangularMeshHydro(peatland=peatland,
                                    peat_hydro_params=peat_hydro_params,
                                    channel_network=channel_network,
                                    cwl_params=cwl_params,
                                    model_coupling=model_coupling,
                                    use_scaled_pde=use_scaled_pde,
                                    zeta_diri_bc=zeta_diri_bc,
                                    force_ponding_storage_equal_one=force_ponding_storage_equal_one)
    else:
        if use_scaled_pde:
            raise Warning('Scaled PDE version is only supported with gmsh mesh')
        return GmshMeshHydro(mesh_fn=mesh_fn,
                             peatland=peatland,
                             peat_hydro_params=peat_hydro_params,
                             channel_network=channel_network,
                             cwl_params=cwl_params,
                             model_coupling=model_coupling,
                             use_scaled_pde=use_scaled_pde,
                             zeta_diri_bc=zeta_diri_bc,
                             force_ponding_storage_equal_one=force_ponding_storage_equal_one)
