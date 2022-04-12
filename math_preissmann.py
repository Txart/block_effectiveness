import numpy as np
import pandas as pd
from numba import njit, jit
import scipy.sparse.linalg
import scipy.linalg
#import pypardiso
import networkx as nx
import scipy.sparse.csgraph as SSC
from tqdm import tqdm

import classes
import cwl_utilities


@njit
def create_diag_special_bool_matrix(n_nodes, position_zeroes):
    # Boolean matrices with the shape needed for the  diag of the Jacobian
    # and positions of ones for each one of the entries: j_AA, j_AQ, ...
    M = np.diag(np.ones(shape=n_nodes), k=0)
    M = cwl_utilities.pad_array_by_interweaving(M, position_zeroes)

    return M


@njit
def create_next_special_bool_matrix(cnm, position_zeroes):
    # Similar to diag function above, but for the j+1, or "next" neighbour nodes
    M = cwl_utilities.pad_array_by_interweaving(cnm.T, position_zeroes)
    return M


@njit
def create_diag_special_matrix_from_vector(vector, position_zeroes):
    # Useful for when jacobian entries are not constant
    # Receives a vector, constructs a matrix by repeating it (tilling it)
    # Then, it is masked by the boolean matrix holding the requested shape
    n_nodes = vector.shape[0]
    # tilled = np.tile(vector, (n_nodes, 1)).T Not supported by Numba
    tilled = np.repeat(vector, repeats=n_nodes).reshape(
        n_nodes, n_nodes)  # Alternative to np.tile
    M = np.multiply(tilled, np.diag(np.ones(shape=n_nodes), k=0))
    M = cwl_utilities.pad_array_by_interweaving(M, position_zeroes)
    return M


@njit
def create_next_special_matrix_from_vector(vector, cnm, position_zeroes):
    # Useful for when jacobian entries are not constant
    # Receives a vector, constructs a matrix by repeating it (tilling it)
    # Then, it is masked by the boolean matrix holding the requested shape
    n_nodes = vector.shape[0]
    # tilled = np.tile(vector, (n_nodes, 1)).T Not supported by Numba
    tilled = np.repeat(vector, repeats=n_nodes).reshape(
        n_nodes, n_nodes)  # Alternative to np.tile
    M = np.multiply(tilled, cnm.T)
    M = cwl_utilities.pad_array_by_interweaving(M, position_zeroes)
    return M

@njit
def variable_n_manning(y: np.ndarray, y_porous_threshold: np.ndarray, n1: float, n2: float) -> np.ndarray:
    # In traditional open channel flow, equations stop being valid when water is below
    # the channel bottom. In reality, though, water still flows in peat
    # (moreover, distinction between peat and channel might be difficult to do)
    # In order to fix that, we introduce the following non-linearity in n_manning:
    # With CWL below n1 given "porous threshold",  y < y_porous_threshold,
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

# %% Cunge-Preissmann
# All the functions that have the word cunge in them
# refer to my second version of the Preissmann scheme
# with y and Q as variables, and the discretization
# given beginning in Cunge (3.81).


@njit
def preissmann_general_term(theta, f, f_next, f_previous, f_previous_next):
    # eq (3.84) in cunge. Approximation of a general term in the Preissmann scheme
    return theta*(f_next + f) + (1-theta)*(f_previous_next + f_previous)


@njit
def preissmann_spatial_derivative(theta, f, f_next, f_previous, f_previous_next):
    # From Eq 3.83 in Cunge
    return theta*(f_next-f) + (1-theta)*(f_previous_next-f_previous)


@njit
def gamma(A, B, n_manning):
    # gamma friction term from notes. Gamma=1/K from Cunge
    return n_manning**2*(2*A/B+B)**(4/3)/A**(7/3)

# Before correcting for y*B -> (y-bottom)*B = A
# @njit
# def gamma_prime(y, B, n_manning):
#     # d(gamma)/dy, where gamma is the friction term in my notes
#     return -n_manning**2 * B*(B+2*y)**(1/3)*(7*B+6*y)/(3*(B*y)**(10/3))

# After correcting for y*B -> (y-bottom)*B = A
@njit
def gamma_prime(A, B, n_manning):
    # d(gamma)/dy, where gamma is the friction term in my notes
    return -n_manning**2 * (B+2*A/B)**(1/3)*(7*B**2+6*A)/(3*A**(10/3)*B)

@njit
def A_wetted_area(y, bottom, B):
    return B*(y - bottom)


@njit
def build_cunge_general_jacobian(y, y_previous, Q, Q_previous, bottom, B, a, g, cnm, dt, dx, n_manning):
    y_next = cnm.T @ y  # A_next means A_{j+1} in Liu and Hodges
    Q_next = cnm.T @ Q
    y_previous_next = cnm.T @ y_previous  # Stands for A^n_{j+1}
    Q_previous_next = cnm.T @ Q_previous
    B_next = cnm.T @ B
    bottom_next = cnm.T @ bottom
    n_nodes = y.shape[0]
    
    A = A_wetted_area(y, bottom, B)
    A_next = A_wetted_area(y_next, bottom_next, B_next)
    A_previous = A_wetted_area(y_previous, bottom, B)
    A_previous_next = A_wetted_area(y_previous_next, bottom_next, B_next)

    term_A = preissmann_general_term(
        a, A, A_next, A_previous, A_previous_next)
    term_Q_abs_Q = preissmann_general_term(a, Q*np.abs(Q), Q_next*np.abs(
        Q_next), Q_previous*np.abs(Q_previous), Q_previous_next*np.abs(Q_previous_next))
    term_gamma = preissmann_general_term(a, gamma(A, B, n_manning), gamma(A_next, B_next, n_manning), gamma(
        A_previous, B, n_manning), gamma(A_previous_next, B_next, n_manning))
    term_B = B_next + B

    #np.seterr(divide='ignore', invalid='ignore')
    # with np.errstate(divide='ignore', invalid='ignore'): # numba doesnt like context managers...
    j_yy = create_diag_special_bool_matrix(
        n_nodes, position_zeroes=('left', 'top'))

    j_yQ = -4*dt*a/dx/term_B
    j_yQ = create_diag_special_matrix_from_vector(
        j_yQ, position_zeroes=('right', 'top'))

    j_Qy = 2*dt/dx*a*Q**2*B/(A**2) + g*dt*a/dx*(-term_A + B_next*(a*(y_next-y) + (1-a) * (
        y_previous_next-y_previous))) + g*dt*0.5*a*term_Q_abs_Q*gamma_prime(A, B, n_manning)
    j_Qy = create_diag_special_matrix_from_vector(
        j_Qy, position_zeroes=('left', 'bottom'))

    j_QQ = 1 - 4*dt/dx*a*Q/A + g*dt*a*term_gamma*np.abs(Q)
    j_QQ = create_diag_special_matrix_from_vector(
        j_QQ, position_zeroes=('right', 'bottom'))

    j_yynext = create_next_special_bool_matrix(
        cnm, position_zeroes=('left', 'top'))

    j_yQnext = 4*dt*a/dx/term_B
    j_yQnext = create_next_special_matrix_from_vector(
        j_yQnext, cnm, position_zeroes=('right', 'top'))

    j_Qynext = -2*dt/dx*a*Q_next**2*B_next/A_next**2 + g*dt/dx*a*(term_A + B_next*(a*(y_next-y) + (
        1-a)*(y_previous_next-y_previous))) + g*dt*0.5*a*term_Q_abs_Q*gamma_prime(A_next, B_next, n_manning)
    j_Qynext = create_next_special_matrix_from_vector(
        j_Qynext, cnm, position_zeroes=('left', 'bottom'))

    j_QQnext = 1 + 4*dt/dx*a*Q_next/A_next  \
        + g*dt*a*term_gamma*np.abs(Q_next)
    j_QQnext = create_next_special_matrix_from_vector(
        j_QQnext, cnm, position_zeroes=('right', 'bottom'))

    # The [:] is a trick I found to make Numba evaluate the array in the moment,
    # instead of lazily evaluating it, which gave problems.
    # Maybe this henders performance?
    jacobian = j_yy[:] + j_yQ[:] + j_Qy[:] + j_QQ[:] + \
        j_yynext[:] + j_yQnext[:] + j_Qynext[:] + j_QQnext[:]

    return jacobian


@njit
def build_general_cunge_F(y, y_previous, Q, Q_previous, q, q_previous, B, a, g, cnm, dt, dx, n_manning):
    y_next = cnm.T @ y  # A_next means A_{j+1} in Liu and Hodges
    Q_next = cnm.T @ Q
    y_previous_next = cnm.T @ y_previous  # Stands for A^n_{j+1}
    Q_previous_next = cnm.T @ Q_previous
    B_next = cnm.T @ B
    q_next = cnm.T @ q
    q_previous_next = cnm.T @ q_previous

    term_A = preissmann_general_term(
        a, B*y, B_next*y_next, B*y_previous, B_next*y_previous_next)
    term_Q_abs_Q = preissmann_general_term(a, Q*np.abs(Q), Q_next*np.abs(
        Q_next), Q_previous*np.abs(Q_previous), Q_previous_next*np.abs(Q_previous_next))
    term_gamma = preissmann_general_term(a, gamma(y, B, n_manning), gamma(y_next, B_next, n_manning), gamma(
        y_previous, B, n_manning), gamma(y_previous_next, B_next, n_manning))
    term_B = B_next + B

    #np.seterr(divide='ignore', invalid='ignore')
    # with np.errstate(divide='ignore', invalid='ignore'):
    F_y = y_next-y_previous_next + y-y_previous + 4*dt/dx*preissmann_spatial_derivative(
        a, Q, Q_next, Q_previous, Q_previous_next)/term_B - dt*preissmann_general_term(a, q, q_next, q_previous, q_previous_next)

    F_Q = Q_next-Q_previous_next + Q - Q_previous + 2*dt/dx*preissmann_spatial_derivative(a, Q**2/(B*y), Q_next**2/(B_next*y_next), Q_previous**2/(B*y_previous), Q_previous_next**2/(
        B_next*y_previous_next)) + g*dt/dx*term_A*preissmann_spatial_derivative(a, y, y_next, y_previous, y_previous_next) + g*dt*0.5*term_Q_abs_Q*term_gamma

    # Interweave to get F_u
    F_u = cwl_utilities.interweave_vectors(F_Q, F_y)

    return F_u


@njit
def fix_BC_in_J(J, pos_to_add_BC_eqs, upstream_nodes, downstream_nodes):
    eq_pos_index = 0
    # Fixed Q upstream
    for up_node in upstream_nodes:
        J[pos_to_add_BC_eqs[eq_pos_index], 2*up_node] = 1.0
        eq_pos_index = eq_pos_index + 1

    # Fixed Y downstream
    for down_node in downstream_nodes:
        J[pos_to_add_BC_eqs[eq_pos_index], 2*down_node+1] = 1.0
        eq_pos_index = eq_pos_index + 1

    return J


@njit
def fix_BC_in_F(F_u):

    # Nothing to be done:
    # Applying fixed BC conds at any node means F[node] = 0
    # But we already ensured that when we "eliminated" the general eqs

    return F_u


def fix_cunge_junctions_in_J(junctions, pos_to_add_junction_eqs, J):
    eq_pos_index = 0
    for up_nodes, down_nodes in junctions:
        # 1. Flow continuity
        for down_node in down_nodes:
            J[pos_to_add_junction_eqs[eq_pos_index], 2*down_node] = 1.0
        for up_node in up_nodes:
            J[pos_to_add_junction_eqs[eq_pos_index], 2*up_node] = -1.0
        eq_pos_index = eq_pos_index + 1

        # 2. The k-1 Y=Y equations in the remaining 1...k-1 positions
        all_nodes = up_nodes + down_nodes
        for i in range(1, len(all_nodes)):
            J[pos_to_add_junction_eqs[eq_pos_index], 2*all_nodes[0]+1] = -1.0
            J[pos_to_add_junction_eqs[eq_pos_index], 2*all_nodes[i]+1] = 1.0
            eq_pos_index = eq_pos_index + 1

    return J


def fix_cunge_junctions_in_F(junctions, pos_to_add_junction_eqs, F_u, y, Q):
    eq_pos_index = 0
    for up_nodes, down_nodes in junctions:
        # 1. Flow continuity equation
        upstream_flow = 0
        for up_node in up_nodes:
            upstream_flow = upstream_flow + Q[up_node]
        downstream_flow = 0
        for down_node in down_nodes:
            downstream_flow = downstream_flow + Q[down_node]

        F_u[pos_to_add_junction_eqs[eq_pos_index]
            ] = downstream_flow - upstream_flow
        eq_pos_index = eq_pos_index + 1

        # The k-1 water height equalities
        all_nodes = up_nodes + down_nodes
        for i in range(1, len(all_nodes)):
            # The first node in all_nodes is the reference. This must be the same in J.
            F_u[pos_to_add_junction_eqs[eq_pos_index]
                ] = y[all_nodes[i]] - y[all_nodes[0]]
            eq_pos_index = eq_pos_index + 1

    return F_u


@njit
def fix_cunge_block_BC_in_J(J, y, g, block_nodes, block_heights, k):
    if len(block_nodes) != 0:
        # Eq. 8.101 in Szymkiewicz's book
        for i in range(len(block_nodes)):
            block_node = block_nodes[i]
            block_height = block_heights[i]
            # Q=0 until it overflows block. Eliminates issues with negative under the square roots
            # The -1 in y[block_node-1] is for y to be taken just *before* the block, not after it.
            parenthesis_term = max(y[block_node-1] - block_height, 0)

            J[2*block_node] = np.zeros(shape=J[0].shape)
            J[2*block_node, 2*block_node] = 1
            J[2*block_node, 2*block_node+1] = -k * \
                1.5 * (2*g*parenthesis_term)**0.5

    return J


@njit
def fix_cunge_block_BC_in_F(F_u, y, Q, g, block_nodes, block_heights, k):
    if len(block_nodes) != 0:
        # Eq. 8.101 in Szymkiewicz's book
        for i in range(len(block_nodes)):
            block_node = block_nodes[i]
            block_height = block_heights[i]
            # Q=0 until it overflows block. Eliminates issues with negative under the square roots
            # The -1 in y[block_node-1] is for y to be taken just *before* the block, not after it.
            parenthesis_term = max(y[block_node-1] - block_height, 0)

            F_u[2*block_node] = Q[block_node] - \
                k * (2*g)**0.5*parenthesis_term**1.5
    return F_u


def remove_equations_from_J(J, eqs_to_remove):
    for eq_position in eqs_to_remove:
        J[eq_position] = np.zeros(shape=(1, J[0].shape[0]))

    return J


def remove_equations_from_F(F_u, eqs_to_remove):
    for eq_position in eqs_to_remove:
        F_u[eq_position] = 0

    return F_u


def build_jacobian(y, y_previous, Q, Q_previous, bottom, B, general_params, channel_network):
    dt = general_params.dt
    dx = general_params.dx
    a = general_params.a
    g = general_params.g
    y_porous_threshold = channel_network.dem - general_params.porous_threshold_below_dem
    n_manning = variable_n_manning(y, y_porous_threshold, n1=general_params.n1, n2=general_params.n2)
    cnm = channel_network.cnm.toarray().astype('float64')
    junctions = channel_network.junctions
    upstream_nodes = channel_network.upstream_nodes
    downstream_nodes = channel_network.downstream_nodes
    block_nodes = channel_network.block_nodes
    block_heights = channel_network.block_heights
    k = channel_network.block_coeff_k

    jacobian = build_cunge_general_jacobian(
        y, y_previous, Q, Q_previous, bottom, B, a, g, cnm, dt, dx, n_manning)
    jacobian = remove_equations_from_J(
        jacobian, channel_network.pos_eqs_to_remove)
    jacobian = fix_cunge_junctions_in_J(junctions, np.array(
        channel_network.pos_of_junction_eqs_to_add), jacobian)
    jacobian = fix_BC_in_J(jacobian, np.array(
        channel_network.pos_of_BC_eqs_to_add), upstream_nodes, downstream_nodes)

    if len(block_nodes) != 0:
        jacobian = fix_cunge_block_BC_in_J(
            jacobian, y, g, block_nodes, block_heights, k)

    return jacobian


def build_F(y, y_previous, Q, Q_previous, q, q_previous, B, general_params, channel_network):
    dt = general_params.dt
    dx = general_params.dx
    a = general_params.a
    g = general_params.g
    y_porous_threshold = channel_network.dem - general_params.porous_threshold_below_dem
    n_manning = variable_n_manning(y, y_porous_threshold, n1=general_params.n1, n2=general_params.n2)
    cnm = channel_network.cnm.toarray().astype('float64')
    junctions = channel_network.junctions
    block_nodes = channel_network.block_nodes
    block_heights = channel_network.block_heights
    k = channel_network.block_coeff_k

    F_u = build_general_cunge_F(
        y, y_previous, Q, Q_previous, q, q_previous, B, a, g, cnm, dt, dx, n_manning)
    F_u = remove_equations_from_F(
        F_u, channel_network.pos_eqs_to_remove)
    F_u = fix_cunge_junctions_in_F(
        junctions, channel_network.pos_of_junction_eqs_to_add, F_u, y, Q)
    F_u = fix_BC_in_F(F_u)
    if len(block_nodes) != 0:
        F_u = fix_cunge_block_BC_in_F(
            F_u, y, Q, g, np.array(block_nodes), np.array(block_heights), k)

    return F_u


@njit
def solve_linear_system(J, F_u):
    return np.linalg.solve(J, -F_u)


def solve_sparse_linear_system(J, F_u):
    return scipy.sparse.linalg.spsolve(scipy.sparse.csr_matrix(J), -F_u)


# %% Steady state

@njit
def build_general_term_SS_jacobian(y, Q, B,  g, cnm, dx, n_manning):
    y_next = cnm.T @ y  # A_next means A_{j+1} in Liu and Hodges
    Q_next = cnm.T @ Q
    B_next = cnm.T @ B
    n_nodes = y.shape[0]

    j_yQ = create_diag_special_bool_matrix(
        n_nodes, position_zeroes=('right', 'top'))

    j_Qy = -Q**2/(B*y**2) + 0.5*g*(2*B*y + B_next*y_next - B*y_next) + \
        0.5*dx*g*Q*np.abs(Q)*gamma_prime(y, B, n_manning)
    j_Qy = create_diag_special_matrix_from_vector(
        j_Qy, position_zeroes=('left', 'bottom'))

    j_QQ = 2*Q/(B*y) + dx*g*np.abs(Q)*gamma(y, B, n_manning)
    j_QQ = create_diag_special_matrix_from_vector(
        j_QQ, position_zeroes=('right', 'bottom'))

    j_yQnext = -create_next_special_bool_matrix(
        cnm, position_zeroes=('right', 'top'))[:]  # Weirdly needed as below for Numba to work

    j_Qynext = Q_next**2/(B_next*y_next**2) + 0.5*g*(-B*y - 2*B_next*y_next + B_next*y) + \
        0.5*dx*g*Q_next*np.abs(Q_next)*gamma_prime(y_next, B_next, n_manning)
    j_Qynext = create_next_special_matrix_from_vector(
        j_Qynext, cnm, position_zeroes=('left', 'bottom'))

    j_QQnext = -2*Q_next/(B_next*y_next) + dx*g * \
        np.abs(Q_next)*gamma(y_next, B_next, n_manning)
    j_QQnext = create_next_special_matrix_from_vector(
        j_QQnext, cnm, position_zeroes=('right', 'bottom'))

    # The [:] is a trick I found to make Numba evaluate the array in the moment,
    # instead of lazily evaluating it, which gave problems.
    # Maybe this henders performance?
    jacobian = j_yQ[:] + j_Qy[:] + j_QQ[:] + \
        j_yQnext[:] + j_Qynext[:] + j_QQnext[:]

    return jacobian


def build_SS_jacobian(y, Q, B, general_params, channel_network):
    cnm = channel_network.cnm.toarray().astype('float64')
    dx = general_params.dx
    g = general_params.g
    n_manning = general_params.n_manning
    junctions = channel_network.junctions
    upstream_nodes = channel_network.upstream_nodes
    downstream_nodes = channel_network.downstream_nodes
    block_nodes = channel_network.block_nodes
    block_heights = channel_network.block_heights
    k = channel_network.block_coeff_k

    jacobian = build_general_term_SS_jacobian(y, Q, B,  g, cnm, dx, n_manning)
    jacobian = remove_equations_from_J(
        jacobian, channel_network.pos_eqs_to_remove)
    jacobian = fix_cunge_junctions_in_J(junctions, np.array(
        channel_network.pos_of_junction_eqs_to_add), jacobian)
    jacobian = fix_BC_in_J(jacobian, np.array(
        channel_network.pos_of_BC_eqs_to_add), upstream_nodes, downstream_nodes)

    if len(block_nodes) != 0:
        jacobian = fix_cunge_block_BC_in_J(
            jacobian, y, g, np.array(block_nodes), np.array(block_heights), k)

    return jacobian


@njit
def build_general_term_SS_F(y, Q, q, B, g, cnm, dx, n_manning):
    y_next = cnm.T @ y  # y_next means y_{j+1} where i = spatial index
    Q_next = cnm.T @ Q
    B_next = cnm.T @ B
    q_next = cnm.T @ q

    F_y = Q - Q_next - 0.5*dx*(B*q + B_next*q_next)
    F_Q = Q**2/(B*y) - Q_next**2/(B_next*y_next) + 0.5*g*(B*y + B_next*y_next)*(y - y_next) + 0.5*dx * \
        g*(Q*np.abs(Q)*gamma(y, B, n_manning) + Q_next *
           np.abs(Q_next)*gamma(y_next, B_next, n_manning))

    # Interweave to get F_u
    F_u = cwl_utilities.interweave_vectors(F_Q, F_y)

    return F_u


def build_SS_F(y, Q, q, B, general_params, channel_network):
    cnm = channel_network.cnm.toarray().astype('float64')
    dx = general_params.dx
    g = general_params.g
    n_manning = general_params.n_manning
    junctions = channel_network.junctions
    block_nodes = channel_network.block_nodes
    block_heights = channel_network.block_heights
    k = channel_network.block_coeff_k

    F_u = build_general_term_SS_F(y, Q, q, B, g, cnm, dx, n_manning)
    F_u = remove_equations_from_F(
        F_u, channel_network.pos_eqs_to_remove)
    F_u = fix_cunge_junctions_in_F(
        junctions, channel_network.pos_of_junction_eqs_to_add, F_u, y, Q)
    F_u = fix_BC_in_F(F_u)
    if len(block_nodes) != 0:
        F_u = fix_cunge_block_BC_in_F(
            F_u, y, Q, g, np.array(block_nodes), np.array(block_heights), k)

    return F_u


def cunge_inexact_newtonRaphson(y, y_previous, Q, Q_previous, q, q_previous, bottom, B, general_params, channel_network, verbose=False, do_force_convergence=True):

    inexact_iter_counter = 0
    compute_and_factorize_jacobian = True
    has_converged = False

    for i in range(general_params.max_niter_newton):
        norm_of_the_previous_solution = np.inf

        if compute_and_factorize_jacobian:
            jacobian = build_jacobian(
                y, y_previous, Q, Q_previous, bottom, B, general_params, channel_network)
            if np.any(~np.any(jacobian, axis=1)):
                raise ValueError(
                    ' The jacobian has at least one row of all zeroes!')
            if np.any(np.isnan(jacobian)):
                raise ValueError('The jacobian has some NaN entry')
            #LU, piv = scipy.linalg.lu_factor(jacobian)
            LU = scipy.sparse.linalg.splu(
                scipy.sparse.csc_matrix(jacobian))  # sparse
            # PYPA_SOLVER = pypardiso.factorized(scipy.sparse.csr_matrix(jacobian)) # PyPARDISO

            compute_and_factorize_jacobian = False
            inexact_iter_counter = 0

        F_u = build_F(
            y, y_previous, Q, Q_previous, q, q_previous, B, general_params, channel_network)

        #x = scipy.linalg.lu_solve((LU, piv), -F_u)
        x = LU.solve(-F_u)
        #x = PYPA_SOLVER(-F_u)

        # The solution is diverging from goal
        if np.linalg.norm(x) > norm_of_the_previous_solution or inexact_iter_counter > general_params.max_niter_inexact:
            compute_and_factorize_jacobian = True

        x_Q, x_y = cwl_utilities.deinterweave_array(x)

        y = y + general_params.weight_A*x_y
        Q = Q + general_params.weight_Q*x_Q

        norm_of_the_previous_solution = np.linalg.norm(x)
        inexact_iter_counter += 1

        if np.linalg.norm(x) < general_params.rel_tol*np.linalg.norm(cwl_utilities.interweave_vectors(y_previous, Q_previous)) + general_params.abs_tol:
            if verbose:
                has_converged = True
                print('\n>>> Inexact Newton-Rhapson converged after ', i, ' iterations')
            return y, Q
        elif np.any(np.isnan(y) | np.isnan(Q)):
            print(
                '\n>>> y: ', y, ' x_y: ', x_y, ' \n>>> y_previous: ', y_previous, ' \n >>> Q: ', Q, ' x_Q: ', x_Q, ' \n>>> Q_previous: ', Q)
            raise ValueError('Nan at some point of Inexact Newton-Raphson')

    if do_force_convergence and not has_converged:
        raise ValueError('The Inexact Newton-Raphson method did not converge in the given iterations, \n and the enforcer do_force_convergence was set to True (it is True by default')
        

    return y, Q


def simulate_one_component(general_params, channel_network):
    cuthill_mckee_permutation = False
    if cuthill_mckee_permutation:
        rcm = SSC.reverse_cuthill_mckee(scipy.sparse.csc_matrix(channel_network.cnm))
        reverse_permutation = np.argsort(rcm)
        cnm = cwl_utilities.permute_row_columns(rcm, channel_network.cnm)

    B = channel_network.B   
    # Initial conditions
    Y_ini = channel_network.y.astype(np.float64)
    Q_ini = channel_network.Q.astype(np.float64)
    bottom = channel_network.bottom.astype(np.float64)
    q = channel_network.q.copy()
    
    if cuthill_mckee_permutation:
        Y_ini = cwl_utilities.permute_vector(rcm, Y_ini)
        Q_ini = cwl_utilities.permute_vector(rcm, Q_ini)

    # Inexact Cunge Jacobian solution
    # In order to change this into exact Jacobian, just make sure
    # compute_and_factorize_jacobian is always set to True

    Q = Q_ini.copy()
    Q_previous = Q_ini.copy()
    y = Y_ini.copy()
    y_previous = Y_ini.copy()
    

    for timestep in range(general_params.ntimesteps):
        y, Q = cunge_inexact_newtonRaphson(
            y, y_previous, Q, Q_previous, q, q, bottom, B, general_params, channel_network)
        y_previous = y.copy()
        Q_previous = Q.copy()

    # Store reesults of last timestep simulation (the rest are not stored)
    if cuthill_mckee_permutation:
        y = cwl_utilities.permute_vector(reverse_permutation, y)
        Q = cwl_utilities.permute_vector(reverse_permutation, Q)

    return y, Q

def simulate_one_component_several_iter(NDAYS, channel_network, general_params):        
    # create this comonent's solution dataframe
    df_y = pd.DataFrame(index=channel_network.graph.nodes)
    df_Q = pd.DataFrame(index=channel_network.graph.nodes)
    # store initial values
    df_y[0] = pd.Series(channel_network.y)
    df_Q[0] = pd.Series(channel_network.Q)
    df_y['DEM'] = pd.Series(channel_network.dem)

    for nday in range(1, NDAYS+1):
        # Simulate
        ysol, Qsol =simulate_one_component(
            general_params, channel_network)

        # update next iteration's initial condition
        channel_network.y = ysol
        channel_network.Q = Qsol

        # Append results
        df_y[nday] = pd.Series(channel_network.from_nparray_to_nodedict(ysol))
        df_Q[nday] = pd.Series(channel_network.from_nparray_to_nodedict(Qsol))

    return df_y, df_Q

def simulate_one_component_SS(general_params, channel_network):
    
    B = channel_network.B
        
    # Initial conditions
    Y_ini = channel_network.y
    Q_ini = channel_network.Q
    q = channel_network.q.copy()
    
    cnm_simple = SS_math_preissmann.simplify_graph_by_removing_junctions(
        channel_network)

    g_simple = nx.DiGraph(incoming_graph_data=cnm_simple.T)

    branches = [g_simple.subgraph(c).copy() for c in sorted(
        nx.weakly_connected_components(g_simple), key=len, reverse=True)]
    length_branches = [len(c) for c in sorted(
        nx.weakly_connected_components(g_simple), key=len, reverse=True)]
    print(
        f'There are {length_branches.count(1)} isolated branches as a result of the junction prunning')

    y_guess = Y_ini.copy()  # Initial guesses for Steady state computation
    Q_guess = Q_ini.copy()
    q_guess = q.copy()

    for branch in tqdm(branches):
        branch_nodes = list(branch.nodes)
        if len(branch_nodes) > 2:  # Isolated nodes are removed from steady state computation
            cnp_branch = classes.ChannelNetworkParameters(nx.adjacency_matrix(
                branch).toarray().T, np.array([]), np.array([]), channel_network.block_coeff_k)
            
            y_branch, Q_branch = SS_math_preissmann.SS_computation(
                y_guess[branch_nodes], Q_guess[branch_nodes], B[branch_nodes], q_guess[branch_nodes], general_params, cnp_branch)

            # update Y_guess
            y_guess[branch_nodes] = y_branch
            Q_guess[branch_nodes] = Q_branch
            
    return y_guess, Q_guess