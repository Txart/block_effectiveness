import numpy as np
import networkx as nx
import pandas as pd
from pathlib import Path

import cwl_utilities

class ChannelNetwork:
    def __init__(self, graph, block_height_from_surface,
                 block_coeff_k, y_ini_below_DEM, y_BC_below_DEM,
                 Q_ini_value, Q_BC, channel_width,
                 channel_bottom_below_DEM,
                 work_without_blocks,
                 is_components=False) -> None:
        """This class has 2 purposes:
                1. get data from cwl network and store it as (properly ordered) nparrays
                2. Compute all things related to the canal topology and its impact to the discrete form of the PDE equations

        Args:
            graph ([type]): [description]
            block_height_from_surface ([type]): [description]
            block_coeff_k ([type]): [description] 
            y_ini_below_DEM ([type]): [description]
            Q_ini_value ([type]): [description]
            q_value ([type]): [description]
            channel_bottom_below_DEM : in m
            work_without_blocks (bool) : if True, run as if there were no blocks in the channel network
            is_components ([bool, default False]) : set some attributes needed to run the diffusive wave
                                                    approximation which are expensive to compute for the whole netwok
                                                    and should only be called for each of the components.
        """
        # Unpack
        self.graph = graph
        self.block_height_from_surface = block_height_from_surface
        self.block_coeff_k = block_coeff_k
        self.y_ini_below_DEM = y_ini_below_DEM  # m below DEM
        self.y_BC_below_DEM = y_BC_below_DEM  # downstream BC (m below DEM)
        self.Q_ini_value = Q_ini_value
        self.Q_BC = Q_BC  # upstream BC in [units of dx]/ [units of dt]
        self.channel_width = channel_width  # (rectangular) canal width in m
        # m below DEM, positive downwards
        self.channel_bottom_below_DEM = channel_bottom_below_DEM
        self.work_without_blocks = work_without_blocks

        # transpose is needed because of difference with NetworkX
        # canal network adjacency matrix
        self.cnm = nx.adjacency_matrix(graph).T
        self.n_nodes = len(graph.nodes)

        # In order to translate labels in the graph to positions in the arrays,
        # nodelist keeps track of position array <--> node label in graph
        # Also worth mentioning: graph.nodes is the default unpacking order used by networkx
        self.nodelist = list(graph.nodes)
        # With this setup, the switch between array coords and graph labels works as follows:
        # 1) graph values -> array: use the method from_graph_attribute_to_nparray()
        # 2) node dictionary -> array: use from_nodedict_to_nparray()
        # 3) array -> node dictionary: use the method from_nparray_to_nodedict()
        # 4) node number -> position in array: use the method translate_from_label_in_graph_to_position_in_vector()

        # change labels in dem dictionary of nodes to positions in the array.
        self.dem = self.from_graph_attribute_to_nparray('DEM')
        self.q = self.from_graph_attribute_to_nparray('q')

        # Topology
        # upstream_bool, downstream_bool = cwl_utilities.infer_extreme_nodes(
        #     self.cnm)  # BC nodes
        upstream_nodes, downstream_nodes = cwl_utilities.infer_extreme_nodes_with_networkx(self.graph)
        # Translate up and downstream nodes to positions in array
        self.downstream_nodes = self.translate_from_label_in_graph_to_position_in_vector(downstream_nodes)
        self.upstream_nodes = self.translate_from_label_in_graph_to_position_in_vector(upstream_nodes)

        # self.downstream_nodes = tuple(np.argwhere(downstream_bool).flatten())
        
        # self.junctions = cwl_utilities.sparse_get_all_junctions(self.cnm)
        self.junctions = cwl_utilities.get_all_junctions_with_networkx(graph)

        if is_components:
            self.undirected_neighbours = self.get_undirected_node_neighbors()

        # Stuff for the system of equations in the Jacobian and F
        self.pos_eqs_to_remove = self.get_positions_of_eqs_to_remove()
        self.number_junction_eqs_to_add = self.count_total_junction_branches()
        # self.check_equal_n_eqs_add_and_remove()
        # These go on top
        self.pos_of_junction_eqs_to_add = self.pos_eqs_to_remove[0:self.number_junction_eqs_to_add]
        self.pos_of_BC_eqs_to_add = self.pos_eqs_to_remove[self.number_junction_eqs_to_add:]

        # Node variables in nparray format
        self.B = channel_width * np.ones(self.n_nodes)
        self.y = self.dem - y_ini_below_DEM
        self.Q = Q_ini_value * np.ones(self.n_nodes)
        # channel bottom from reference datum, m
        self.bottom = self.dem - channel_bottom_below_DEM

        # Set y_BC and Q_BC inside the initial conditions
        # dtype=int necessary in case downstream_nodes or upstream_nodes are empty lists
        self.y[np.array(self.downstream_nodes, dtype=int)] = self.dem[np.array(self.downstream_nodes, dtype=int)] - \
            self.y_BC_below_DEM
        self.Q[np.array(self.upstream_nodes, dtype=int)] = self.Q_BC

        # Block stuff
        if work_without_blocks:
            self.block_nodes = []
            self.block_heights = np.array([])
            self.outgoing_neighbors_of_blocks = []
            
        elif work_without_blocks == False:
            block_attr = nx.get_node_attributes(self.graph, 'is_block')
            nodes_with_blocks = [node for node, is_block in block_attr.items() if is_block]
            self.block_nodes = self.translate_from_label_in_graph_to_position_in_vector(nodes_with_blocks)
            self.block_heights = np.array(
                [self.dem[n] + block_height_from_surface for i, n in enumerate(self.block_nodes)])
            
            # Manage blocks and neighboring nodes for diff wave
            if is_components:
                self.outgoing_neighbors_of_blocks = []
                for n in nodes_with_blocks:
                    outgoing_from_n = self.translate_from_label_in_graph_to_position_in_vector(list(self.graph.successors(n)))
                    if len(outgoing_from_n) == 0: # No outgoing neighbours
                        self.outgoing_neighbors_of_blocks.append(-1) # Append a minus one, used in math_diff wave as well
                        continue
                    else:
                        outgoing_from_n = outgoing_from_n[0] # I assume there's only one outgoing neighbor for each block. This simplifies the whole thing!
                        self.outgoing_neighbors_of_blocks.append(outgoing_from_n)
                        n_in_array = self.translate_from_label_in_graph_to_position_in_vector([n])[0]
                        # Remove incoming block neighbors
                        self.undirected_neighbours[outgoing_from_n].remove(n_in_array)
                        # Remove outgoing neighbors of the block node
                        self.undirected_neighbours[n_in_array].remove(outgoing_from_n)
        

        # Ordered list of component graphs
        self.component_graphs = self.find_graph_components()

        pass

    def get_positions_of_eqs_to_remove(self):
        nodes_to_remove = []
        # Junction eqs
        for up_nodes, _ in self.junctions:
            for up_node in up_nodes:
                nodes_to_remove = nodes_to_remove + [2*up_node, 2*up_node+1]
        # BCs
        for down_node in self.downstream_nodes:
            nodes_to_remove = nodes_to_remove + [2*down_node, 2*down_node+1]

        return nodes_to_remove

    def count_total_junction_branches(self):
        count = 0
        for up_nodes, down_nodes in self.junctions:
            count = count + len(up_nodes) + len(down_nodes)
        return count

    def check_equal_n_eqs_add_and_remove(self):
        number_BC_eqs_to_add = len(
            self.downstream_nodes) + len(self.upstream_nodes)
        total_number_eqs_to_add = self.number_junction_eqs_to_add + number_BC_eqs_to_add
        if total_number_eqs_to_add != len(self.pos_eqs_to_remove):
            raise ValueError(
                'Mismatch in number of eqs to add and to remove from J and F.')
        return 0

    def translate_from_label_in_graph_to_position_in_vector(self, list_node_labels):
        # Takes a list of ints specifying the labels of some nodes in the graph
        # and returns their positions in the vectorized of the eqs
        dict_from_label_in_graph_to_position_in_vector = {
            n_node: i for n_node, i in zip(self.nodelist, range(0, self.n_nodes))}
        return [dict_from_label_in_graph_to_position_in_vector[n_node] for n_node in list_node_labels]

    def from_nodedict_to_nparray(self, node_var):
        # Takes variable in nodedict and returns a properly ordered nparray
        return np.array([node_var[n] for n in self.nodelist])

    def from_nparray_to_nodedict(self, var_vector):
        # Takes numpy array that holds the value of a channel network variable and
        # returns it as a dictionary of nodes with original graph labels
        node_var = dict()
        for value, nodename in zip(var_vector, self.nodelist):
            node_var[nodename] = value
        return node_var

    def add_node_attribute_from_nparray(self, values_array, attr_name):
        values_dict = self.from_nparray_to_nodedict(values_array)
        nx.set_node_attributes(self.graph, values=values_dict, name=attr_name)
        return 0

    def from_graph_attribute_to_nparray(self, attr_name):
        attr = nx.get_node_attributes(self.graph, attr_name)
        return np.array(list(attr.values()))

    def find_graph_components(self):
        component_graphs = [self.graph.subgraph(c).copy() for c in sorted(
            nx.weakly_connected_components(self.graph), key=len, reverse=True)]
        # Remove components with fewer than 3 nodes
        return [g for g in component_graphs if len(g.nodes) > 3]

    def plot_graph(self):
        pos = nx.planar_layout(self.graph)
        options = {
            "font_size": 20,
            "linewidths": 5,
            "width": 5,
        }
        nx.draw_networkx(self.graph, pos=pos, **options)
        return None

    def get_undirected_graph(self):
        return self.graph.to_undirected()

    def get_undirected_node_neighbors(self) -> list:
        # Return node numbers in terms of the array positions, not the node labels
        ungraph = self.get_undirected_graph()  # Neighbors from the undirected graph
        nodes_neighbours = [self.translate_from_label_in_graph_to_position_in_vector(
                            list(ungraph.neighbors(i)))
                            for i in self.nodelist]

        return nodes_neighbours
    
    def burn_nodedict_into_graph(self, nodedict, name_of_graph_attribute):
        nx.set_node_attributes(
            self.graph, values=nodedict, name=name_of_graph_attribute)
        return None
    
    def save_graph_attribute_to_geoJSON_file(self, attribute: str, filepath: str, projection='epsg:32748'):
        df_nodes = cwl_utilities.create_dataframe_from_node_attributes(
            self.graph)
        gdf = cwl_utilities.convert_dataframe_to_geodataframe(
            df_nodes, 'x', 'y', projection)

        col_to_export = attribute
        gdf_to_export = gdf[[col_to_export, 'geometry']]

        gdf_to_export.to_file(filepath, driver='GeoJSON')

       return None
