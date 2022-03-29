# Translate from Fortran to Python for less hassles
# This is part of the content of the Fortran file fd-new.f95

#%%
import numpy as np
from numba import njit
from numpy.lib.type_check import real


#%% Peat hydraulic properties

# @njit
# def h_from_zeta(zeta, dem):
#     return zeta + dem

# @njit
# def zeta_from_h(h, dem):
#     return h - dem

# @njit
# def h_from_theta(theta, s2):
#     return np.sqrt(2*theta/s2)

# @njit
# def theta_from_h(h, s2):
#     return 0.5*s2*h**2

# @njit
# def dif(theta, dem, s2, t1, t2):
#     return t1*np.exp(-t2*dem)*(np.exp(t2*np.sqrt(2*theta/s2)) - 1)/ np.sqrt(2*s2*theta)

# @njit
# def dif_prime(theta, dem, s2, t1, t2):
#     return t1*np.exp(-t2*dem)/(2**1.5 * np.sqrt(s2) * theta**1.5) * (np.exp(t2*np.sqrt(2*theta/s2))*(t2*np.sqrt(2*theta/s2) - 1) + 1)


@njit
def h_from_zeta(zeta, dem):
    return zeta + dem

@njit
def zeta_from_h(h, dem):
    return h - dem

@njit
def h_from_theta(theta, dem, s1, s2):
    return np.log(s2/s1 * np.exp(s2*dem) * theta + 1)/s2

@njit
def theta_from_h(h, dem, s1, s2):
    return s1/s2 * (np.exp(s2*(h - dem)) - np.exp(-s2*dem))

# Old dif and dif_prime, with risk of machine precision overflow
# @njit
# def dif(theta, dem, b, s1, s2, t1, t2):
#     return (
#             t1*((((s1 + s2*theta*np.exp(dem*s2))/s1)**t2/s2)*np.exp(b*t2) - np.exp(dem*t2))
#             * np.exp(-b*t2 + dem*s2 - dem*t2)/(s1 + s2*theta*np.exp(dem*s2))
#             )       
# @njit
# def dif_prime(theta, dem, b, s1, s2, t1, t2):
#     return (
#             t1*(-s2*((((s1 + s2*theta*np.exp(dem*s2))/s1)**t2/s2)*np.exp(b*t2) - np.exp(dem*t2))*np.exp(dem*(2*s2 + t2)) +
#                 t2*(((s1 + s2*theta*np.exp(dem*s2))/s1)**t2/s2)*np.exp(b*t2 + 2*dem*s2 + dem*t2))
#             *np.exp(-t2*(b + 2*dem))/(s1 + s2*theta*np.exp(dem*s2))**2
#             )

# New dif and dif_prime, with less machine precision overflow risk
# Otherwise, equivalent
@njit
def dif(theta, dem, b, s1, s2, t1, t2):
    return (
            t1/s1**(t2/s2)*(s1*np.exp(-dem*s2) + s2*theta)**(t2/s2-1) -
                t1*np.exp(-b*t2)/(s1*np.exp(-dem*s2) + s2*theta)
            )       

@njit
def dif_prime(theta, dem, b, s1, s2, t1, t2):
    return (
            t1/s1**(t2/s2)*(t2 - s2)*(s1*np.exp(-dem*s2) + s2*theta)**(t2/s2-2) +
                t1*s2*np.exp(-b*t2)/(s1*np.exp(-dem*s2) + s2*theta)**2
            )


#%% Several algorithms to get J and F for different BC and transient or steadystates

@njit
def subroutine_j_and_f_both_diri(N:int, v:np.ndarray, v_old:np.ndarray, dem:np.ndarray, b:float,
                       delta_t:float, delta_x: float, s1:float, s2: float, t1:float, t2:float, source:float):
    
    #v here stands for theta in the notes
    # b is peat depth. Assumed constant for all transect.
    
    # Notation
    e = 1/(2*delta_x**2)
    
    # Allocate arrays
    J = np.zeros(shape=(N,N))
    F = np.zeros(shape=N)
    
    for i in range(1,N-1):
        J[i,i-1] = e*(dif_prime(v[i-1], dem[i-1], b, s1, s2, t1, t2)*(v[i] - v[i-1]) -
                      dif(v[i], dem[i], b, s1, s2, t1, t2) - dif(v[i-1], dem[i-1], b, s1, s2, t1, t2))

        J[i,i] = e*(dif_prime(v[i], dem[i], b, s1, s2, t1, t2)*(-v[i-1] + 2*v[i] -v[i+1]) +
                    dif(v[i+1], dem[i+1], b, s1, s2, t1, t2) + 
                    2*dif(v[i], dem[i], b, s1, s2, t1, t2) +
                    dif(v[i-1], dem[i-1], b, s1, s2, t1, t2)) + 1/delta_t

        J[i,i+1] = e*(dif_prime(v[i+1], dem[i+1], b, s1, s2, t1, t2)*(v[i] - v[i+1])-
                      dif(v[i+1], dem[i+1], b, s1, s2, t1, t2) - 
                      dif(v[i], dem[i], b, s1, s2, t1, t2))

        # F
        F[i] = -e*((dif(v[i], dem[i], b, s1, s2, t1, t2) + dif(v[i+1], dem[i+1], b, s1, s2, t1, t2))*(v[i+1] - v[i]) -
                   (dif(v[i-1], dem[i-1], b, s1, s2, t1, t2) + dif(v[i], dem[i], b, s1, s2, t1, t2))*(v[i] - v[i-1])) + (v[i]-v_old[i])/delta_t - source
    
    # BC
    # Diri in x=0
    J[0,0] = 1
    F[0] = 0.0
    # Diri in x=N
    J[-1,-1] = 1
    
    F[-1] = 0
    
    return J, F


# %% Full solvers


@njit
def solve_transient_both_diri(theta, dx, dt, NDAYS, N, dem, b, s1, s2, t1, t2,
                             SOURCE, ZETA_LEFT_DIRIS, ZETA_RIGHT_DIRIS,
                             MAX_INTERNAL_NITER, WEIGHT, REL_TOL, ABS_TOL,
                             verbose=False):
    solution_zeta = np.zeros(shape=(NDAYS, N))
    
    for day in range(NDAYS):
        # BC and Source/sink update
        # LEFT DIRI BC
        h_left = h_from_zeta(ZETA_LEFT_DIRIS[day], dem[0])
        theta_left = theta_from_h(h_left, dem[0], s1=s1, s2=s2) # left BC is always Dirichlet. No-flux in the right all the time
        theta[0] = theta_left
        
        # RIGHT DIRI BC
        h_right = h_from_zeta(ZETA_RIGHT_DIRIS[day], dem[-1])
        theta_right = theta_from_h(h_right, dem[-1], s1=s1, s2=s2) # left BC is always Dirichlet. No-flux in the right all the time
        theta[-1] = theta_right
        
        theta_old = theta.copy()

        for i in range(0, MAX_INTERNAL_NITER):
            J, F = subroutine_j_and_f_both_diri(N=N, v=theta, v_old=theta_old, dem=dem, b=b,
                                                        delta_t=dt, delta_x=dx,
                                                        s1=s1, s2=s2, t1=t1, t2=t2, source=SOURCE[day])
            eps_x = np.linalg.solve(J,-F)
            theta = theta + WEIGHT*eps_x

            # stopping criterion
            if np.linalg.norm(eps_x) < REL_TOL*np.linalg.norm(theta) + ABS_TOL:
                break

        # Early stopping criterion: theta cannot be negative
        if np.any(theta < 0) or np.any(np.isnan(theta)):
            
            raise ValueError('NEGATIVE V FOUND, ABORTING')

        if verbose:
            print('\n Number of run internal iterations: ' + str(i))
        
        # Append solution
        solution_h = h_from_theta(theta, dem, s1, s2)
        solution_zeta[day] = zeta_from_h(solution_h, dem)
        
    return solution_zeta
