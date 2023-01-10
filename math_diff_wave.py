import numpy as np
import scipy.linalg
from numba import njit
from numba.typed import List

# %%


@njit
def compute_A(y, y_b, B) -> np.ndarray:
    # y is height of cwl from ref datum
    # y_b is height of canal bottom from ref datum
    return (y - y_b)*B


@njit
def compute_R(y, y_b, B):
    return (y-y_b)*B/(2*(y-y_b)+B)


@njit
def compute_K(A, R, n_manning):
    return A*R**(2/3)/n_manning


@njit
def K_mean(i, j, K):
    return 0.5*(K[i] + K[j])


@njit
def compute_dA_dh(B):
    return 0.5*B


@njit
def compute_dR_dh(y, y_b, B):
    return (B/(2*(y-y_b) + B))**2


@njit
def compute_dK_mean_dh(A, R, dA_dh, dR_dh, n_manning):
    return 0.5/n_manning*(dA_dh*R**(2/3) + 2/3*A/R**(1/3)*dR_dh)


@njit
def variable_n_manning(y: np.ndarray, y_porous_threshold: np.ndarray, n1: float, n2: float) -> np.ndarray:
    # In traditional open channel flow, equations stop being valid when water is below
    # the channel bottom. In reality, though, water still flows in peat
    # (moreover, distinction between peat and channel might be difficult to do)
    # In order to fix that, we introduce the following non-linearity in n_manning:
    # With CWL below a given "porous threshold",  y < y_porous_threshold,
    # friction term is high and constant. This makes the results mimick flow through a porous medium.
    # Above that threshold, the friction term decreases as a power law y**(-n1).
    # n1 is the only free parameter, and regulates how steep the non-linearity is.

    # y, y_b and y_porous_threshold are all CWLs measured from reference datum (m)
    
    # Change of variables to simplify coding. zeta is a local variable and has nothing to do with zeta at peatland hydrology.
    zeta = y - y_porous_threshold
    # Cap at porous threshold
    zeta[zeta < 0] = 0.0
    
    N_POROUS_THRESHOLD = 10000 # value at which n is capped

    n_manning = N_POROUS_THRESHOLD / np.exp(n1 * zeta**n2)

    return n_manning


@njit
def compute_J_and_F(y: np.ndarray, y_old: np.ndarray,
                    y_b: np.ndarray, B: np.ndarray, q: np.ndarray,
                    dt: float, dx: float,
                    y_porous_threshold: np.ndarray, n1: float, n2: float,
                    nodes: np.ndarray, nodes_neighbours: list):
    """computes jacobian and right hand side for newton raphson method

    Args:
        y (np.ndarray): [description]
        y_old (np.ndarray): [description]
        y_b (np.ndarray): [description]
        B (np.ndarray): [description]
        dt (float): [description]
        dx (float): [description]
        n1 and n2: parameters to compute n_manning
        nodes (np.ndarray): Node numbering. Entries correspond to position in matrix
        nodes_neighbours (np.ndarray of np.ndarray): Each entry of the np.array is an np.array that contains
        the number of the (undirected) neighbours of the list of nodes above.  

    Returns:
        [type]: [description]
    """
    # Here, y is canal water height from common reference datum, not from DEM
    # y_b is channel bottom from common ref point
    n_manning = variable_n_manning(y, y_porous_threshold, n1=n1, n2=n2)
    A_vec = compute_A(y, y_b, B)
    R_vec = compute_R(y, y_b, B)
    K_vec = compute_K(A_vec, R_vec, n_manning)
    dA_dh_vec = compute_dA_dh(B)
    dR_dh_vec = compute_dR_dh(y, y_b, B)
    dK_mean_dh_vec = compute_dK_mean_dh(
        A_vec, R_vec, dA_dh_vec, dR_dh_vec, n_manning)

    # scnm = channel_network.cnm + channel_network.cnm.T # symmetric, undirected
    # i,j,_ = scipy.sparse.find(scnm)

    # Compute Jacobian and F
    J = np.diag(np.ones_like(y))  # begin by adding 1 in the diagonal
    F_u = y - y_old - dt*q/B

    for node, neighbors in zip(nodes, nodes_neighbours):
        if neighbors == List([-1]): # No neighbours flag
            continue # cycle the outermost loop for a new neighbour
        else:
            for neig in neighbors:
                h_diff = y[node] - y[neig]
                h_sign = np.sign(h_diff)
                sqrt_h_diff = np.sqrt(np.abs(h_diff))
                alpha = dt/dx**1.5/B[node]

                K_mean_node_neigh = K_mean(node, neig, K_vec)

                # F
                F_u[node] = F_u[node] + alpha * \
                    K_mean_node_neigh * sqrt_h_diff * h_sign

                # Jacobian
                if sqrt_h_diff < 1e-10:  # h_i = h_j condition
                    J[node, node] = 1.
                    J[node, neig] = 0.
                else:
                    J[node, node] = J[node, node] + alpha * \
                        (h_sign*dK_mean_dh_vec[node]*sqrt_h_diff +
                        0.5*K_mean_node_neigh/sqrt_h_diff)
                    J[node, neig] = alpha * (h_sign*dK_mean_dh_vec[neig]
                                            * sqrt_h_diff - 0.5*K_mean_node_neigh/sqrt_h_diff)

    return J, F_u


@njit
def compute_F(y: np.ndarray, y_old: np.ndarray,
              y_b: np.ndarray, B: np.ndarray, q: np.ndarray,
              dt: float, dx: float,
              y_porous_threshold: np.ndarray, n1: float, n2: float,
              nodes: np.ndarray, nodes_neighbours: list):
    """computes only  right hand side (not the Jacobian) for newton raphson method.
    Useful for inexact NR method. 

    Args:
        y (np.ndarray): [description]
        y_old (np.ndarray): [description]
        y_b (np.ndarray): [description]
        B (np.ndarray): [description]
        dt (float): [description]
        dx (float): [description]
        nodes (np.ndarray): Node numbering. Entries correspond to position in matrix
        nodes_neighbours (np.ndarray of np.ndarray): Each entry of the np.array is an np.array that contains
        the number of the (undirected) neighbours of the list of nodes above.  

    Returns:
        [type]: [description]
    """
    # Here, y is canal water height from common reference datum, not from DEM
    # y_b is channel bottom from common ref point
    n_manning = variable_n_manning(y, y_porous_threshold, n1, n2)
    A_vec = compute_A(y, y_b, B)
    R_vec = compute_R(y, y_b, B)
    K_vec = compute_K(A_vec, R_vec, n_manning)

    # F
    F_u = y - y_old - dt*q/B

    for node, neighbors in zip(nodes, nodes_neighbours):
        if neighbors == List([-1]): # no neighbours flag
            continue # cycle the outermost loop
        else:
            for neig in neighbors:
                h_diff = y[node] - y[neig]
                h_sign = np.sign(h_diff)
                sqrt_h_diff = np.sqrt(np.abs(h_diff))
                alpha = dt/dx**1.5/B[node]

                K_mean_node_neigh = K_mean(node, neig, K_vec)

                # F
                F_u[node] = F_u[node] + alpha * \
                    K_mean_node_neigh * sqrt_h_diff * h_sign

    return F_u

@njit
def _solve_newton_raphson_numba(max_niter: int, dt: float, dx: float, g:float, n1: float, n2: float,
                                y_porous_threshold: np.ndarray,
                                y: np.ndarray, y_previous: np.ndarray, bottom: np.ndarray, B: np.ndarray, q: np.ndarray,
                                nodes: np.ndarray, typed_node_neighbours: list,
                                block_heights: np.ndarray,
                                block_positions: np.ndarray,
                                block_coeff_k: float,
                                block_outgoing_neighbours: np.ndarray,
                                downstream_nodes: np.ndarray,
                                weight: float, rel_tol: float, abs_tol: float,
                                downstream_diri_BC:bool,
                                verbose=False):
    for i in range(max_niter):
        J, F_u = compute_J_and_F(
            y=y, y_old=y_previous, y_b=bottom, B=B, q=q,
            dt=dt, dx=dx,
            y_porous_threshold=y_porous_threshold, n1=n1, n2=n2,
            nodes=nodes,
            nodes_neighbours=typed_node_neighbours)
       
        # Block stuff
        if len(block_positions) != 0:
            for n, blockpos in enumerate(block_positions):
                diff = max(y[blockpos] - block_heights[n], 0)
                block_flux = dt/dx*block_coeff_k*np.sqrt(2*g)*diff**1.5
                block_flux_prime = dt/dx*block_coeff_k*np.sqrt(2*g)*1.5*np.sqrt(diff)
                F_u[blockpos] += block_flux
                J[blockpos, blockpos] += block_flux_prime
                out_neig = block_outgoing_neighbours[n]
                if out_neig != -1: # -1 = no outgoing neighbours of a block
                    F_u[out_neig] += -block_flux
                    J[out_neig, blockpos] += -block_flux_prime
                    
         # Downstream fixed cwl BC
         # If False, do nothing, which equals Neumann no-flux BC
         # If True, keep initial value of y at downstream nodes.
        if downstream_diri_BC: 
            for downstream_node in downstream_nodes:
                J[downstream_node] = np.zeros_like(y)
                J[downstream_node, downstream_node] = 1.
                F_u[downstream_node] = 0

        x_y = np.linalg.solve(J, -F_u)
        y = y + weight*x_y

        #print(np.linalg.norm(x_y) - rel_tol*np.linalg.norm(y) + abs_tol)

        if np.linalg.norm(x_y) < rel_tol*np.linalg.norm(y) + abs_tol:
            if verbose:
                print('\n>>> Diff wave Newton-Rhapson converged after ',
                      str(i), ' iterations')
            return y

        elif np.any(np.isnan(y)):
            raise ValueError('Nan at some point of Inexact Newton-Raphson')

    raise RuntimeError(
        'Diff wave Newton-Raphson didnt converge after max number of iterations')


def _solve_newton_raphson_inexact_newton_raphson(max_niter: int, max_niter_inexact: int, 
                                dt: float, dx: float, g:float, n1: float, n2: float,
                                y_porous_threshold: np.ndarray,
                                y: np.ndarray, y_previous: np.ndarray, bottom: np.ndarray, B: np.ndarray, q: np.ndarray,
                                nodes: np.ndarray, typed_node_neighbours: list,
                                block_heights: np.ndarray,
                                block_positions: np.ndarray,
                                block_coeff_k: float,
                                block_outgoing_neighbours: np.ndarray,
                                downstream_nodes: np.ndarray,
                                weight: float, rel_tol: float, abs_tol: float,
                                downstream_diri_BC:bool,
                                verbose=False):
    inexact_iter_counter = 0
    compute_and_factorize_jacobian = True
    has_converged = False
    norm_of_the_previous_solution = np.inf

    for i in range(max_niter):
        if compute_and_factorize_jacobian:
            J, F_u = compute_J_and_F(
                y=y, y_old=y_previous, y_b=bottom, B=B, q=q,
                dt=dt, dx=dx,
                y_porous_threshold=y_porous_threshold, n1=n1, n2=n2,
                nodes=nodes,
                nodes_neighbours=typed_node_neighbours)
            
            # Block stuff
            if len(block_positions) != 0:
                for n, blockpos in enumerate(block_positions):
                    diff = max(y[blockpos] - block_heights[n], 0)
                    block_flux = dt/dx*block_coeff_k*np.sqrt(2*g)*diff**1.5
                    block_flux_prime = dt/dx*block_coeff_k*np.sqrt(2*g)*1.5*np.sqrt(diff)
                    F_u[blockpos] += block_flux
                    J[blockpos, blockpos] += block_flux_prime
                    out_neig = block_outgoing_neighbours[n]
                    if out_neig != -1: # -1 = no outgoing neighbours of a block
                        F_u[out_neig] += -block_flux
                        J[out_neig, blockpos] += -block_flux_prime
            
            # Downstream fixed cwl BC
            # If False, do nothing, which equals Neumann no-flux BC
            # If True, keep initial value of y at downstream nodes.
            if downstream_diri_BC: 
                for downstream_node in downstream_nodes:
                    J[downstream_node] = np.zeros_like(y)
                    J[downstream_node, downstream_node] = 1.
                    F_u[downstream_node] = 0
            
            # LU decomposition
            LU = scipy.sparse.linalg.splu(scipy.sparse.csc_matrix(J))  # sparse

            compute_and_factorize_jacobian = False
            inexact_iter_counter = 0
        else:
            F_u = compute_F(
                y=y, y_old=y_previous, y_b=bottom, B=B, q=q,
                dt=dt, dx=dx,
                y_porous_threshold=y_porous_threshold,  n1=n1, n2=n2,
                nodes=nodes,
                nodes_neighbours=typed_node_neighbours)
            
            # Block stuff
            if len(block_positions) != 0:
                for n, blockpos in enumerate(block_positions):
                    diff = max(y[blockpos] - block_heights[n], 0)
                    block_flux = dt/dx*block_coeff_k*np.sqrt(2*g)*diff**1.5
                    F_u[blockpos] += block_flux
                    out_neig = block_outgoing_neighbours[n]
                    if out_neig != -1: # -1 = no outgoing neighbours of a block
                        F_u[out_neig] += -block_flux
                        
            # Downstream fixed cwl BC
            # If False, do nothing, which equals Neumann no-flux BC
            # If True, keep initial value of y at downstream nodes.
            for downstream_node in downstream_nodes:
                F_u[downstream_node] = 0

        # Solution
        x_y = LU.solve(-F_u)

        # If solution is diverging from goal
        if np.linalg.norm(x_y) > norm_of_the_previous_solution or inexact_iter_counter > max_niter_inexact:
            compute_and_factorize_jacobian = True
            norm_of_the_previous_solution = np.inf

        else:
            y = y + weight*x_y
            norm_of_the_previous_solution = np.linalg.norm(x_y)
            inexact_iter_counter += 1

        if np.linalg.norm(x_y) < rel_tol*np.linalg.norm(y) + abs_tol:
            if verbose:
                print('\n>>> Diff wave Newton-Rhapson converged after ',
                      str(i), ' iterations')
            return y

        elif np.any(np.isnan(y)):
            raise ValueError('Nan at some point of Inexact Newton-Raphson')

    raise RuntimeError(
        'Diff wave Newton-Raphson didnt converge after max number of iterations')

    return None


def diff_wave_newton_raphson(y, y_previous, bottom, B, q, general_params, channel_network):
    # Recommended that this function keeps the same interface as the preissmann function.
    # However, as oposed to the preissmann case, most of the computations can be numbified, and so I do.

    # nodes and node_neighbours must be given in terms of the array position number
    nodes = channel_network.translate_from_label_in_graph_to_position_in_vector(
        channel_network.nodelist)
    node_neighbours = channel_network.undirected_neighbours
    typed_node_neighbours = List()  # needed Typed List for numba function
    for neig_list in node_neighbours:
        typed_neig_list = List()
        for neig in neig_list:
            typed_neig_list.append(neig)
        if len(neig_list) == 0:
            typed_neig_list.append(-1) # When there are no neighbours, append a -1. Numba requires having integers in type integer lists        
        typed_node_neighbours.append(typed_neig_list)
        
    # print('block_nodes : ', channel_network.block_nodes)

    y_sol = _solve_newton_raphson_numba(
        max_niter=general_params.max_niter_newton,
        dt=general_params.dt, dx=general_params.dx,
        g=general_params.g,
        n1=general_params.n1, n2=general_params.n2,
        y_porous_threshold = channel_network.dem - general_params.porous_threshold_below_dem,
        y=y, y_previous=y_previous, bottom=bottom, B=B, q=q,
        nodes=np.array(nodes), typed_node_neighbours=typed_node_neighbours,
        block_heights=channel_network.block_heights,
        block_positions=np.array(channel_network.block_nodes, dtype='int32'),
        block_coeff_k=channel_network.block_coeff_k,
        block_outgoing_neighbours=np.array(channel_network.outgoing_neighbors_of_blocks, dtype='int32'),
        downstream_nodes=np.array(channel_network.downstream_nodes),
        weight=general_params.weight_A,
        rel_tol=general_params.rel_tol, abs_tol=general_params.abs_tol,
        downstream_diri_BC=general_params.downstream_diri_BC,
        verbose=False)

    return y_sol


def diff_wave_inexact_newton_raphson(y, y_previous, bottom, B, q, general_params, channel_network):
    # Recommended that this function keeps the same interface as the preissmann function.
    # However, as oposed to the preissmann case, most of the computations can be numbified, and so I do.

     # nodes and node_neighbours must be given in terms of the array position number
    nodes = channel_network.translate_from_label_in_graph_to_position_in_vector(
        channel_network.nodelist)
    node_neighbours = channel_network.undirected_neighbours
    typed_node_neighbours = List()  # needed Typed List for numba function
    for neig_list in node_neighbours:
        typed_neig_list = List()
        for neig in neig_list:
            typed_neig_list.append(neig)
        if len(neig_list) == 0:
            typed_neig_list.append(-1) # When there are no neighbours, append a -1. Numba requires having integers in type integer lists        
        typed_node_neighbours.append(typed_neig_list)

    y_sol = _solve_newton_raphson_inexact_newton_raphson(
        max_niter=general_params.max_niter_newton,
        max_niter_inexact=general_params.max_niter_inexact,
        dt=general_params.dt, dx=general_params.dx,
        g=general_params.g,
        n1=general_params.n1, n2=general_params.n2,
        y_porous_threshold = channel_network.dem - general_params.porous_threshold_below_dem,
        y=y, y_previous=y_previous, bottom=bottom, B=B, q=q,
        nodes=np.array(nodes), typed_node_neighbours=typed_node_neighbours,
        block_heights=channel_network.block_heights,
        block_positions=np.array(channel_network.block_nodes, dtype='int32'),
        block_coeff_k=channel_network.block_coeff_k,
        block_outgoing_neighbours=np.array(channel_network.outgoing_neighbors_of_blocks, dtype='int32'),
        downstream_nodes=np.array(channel_network.downstream_nodes),
        weight=general_params.weight_A,
        rel_tol=general_params.rel_tol, abs_tol=general_params.abs_tol,
        downstream_diri_BC=general_params.downstream_diri_BC,
        verbose=False)

    return y_sol


def simulate_one_component(general_params, channel_network):
    B = channel_network.B
    q = channel_network.q
    # Initial conditions
    Y_ini = channel_network.y.astype(np.float64)
    bottom = channel_network.bottom.astype(np.float64)

    y = Y_ini.copy()
    y_previous = Y_ini.copy()

    for timestep in range(general_params.ntimesteps):
        y = diff_wave_newton_raphson(
            y, y_previous, bottom, B, q, general_params, channel_network)
        y_previous = y.copy()

    return y


def simulate_one_component_inexact_newton_raphson(general_params, channel_network):
    B = channel_network.B
    q = channel_network.q
    # Initial conditions
    Y_ini = channel_network.y.astype(np.float64)
    bottom = channel_network.bottom.astype(np.float64)

    y = Y_ini.copy()
    y_previous = Y_ini.copy()

    for timestep in range(general_params.ntimesteps):
        y = diff_wave_inexact_newton_raphson(
            y, y_previous, bottom, B, q, general_params, channel_network)
        y_previous = y.copy()

    return y

# %%
