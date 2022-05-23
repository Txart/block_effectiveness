import matplotlib.pyplot as plt
from fipy import numerix


from classes.peatland_hydrology import PeatlandHydroParameters


class AbstractParameterization:
    def __init__(self,
                 ph_params: PeatlandHydroParameters) -> None:

        self.s1 = ph_params.s1
        self.s2 = ph_params.s2
        self.t1 = ph_params.t1
        self.t2 = ph_params.t2

        pass

    """
    This class is abstract. Always create one of its subclasses.
    Here I define 3 functions that every subclass of this class
    should have.
    """

    def theta_from_zeta(self, zeta, dem):
        raise NotImplementedError

    def zeta_from_theta(self, theta, dem):
        raise NotImplementedError

    def diffusion(self, theta, dem, b):
        raise NotImplementedError

    """
    Not necessary for hydro, but needed for plotting.
    Implemented by subclasses.
    """

    def storage_from_zeta(self, zeta, dem):
       return None

    def transmissivity_from_zeta(self, zeta, dem, b):
        return None

    """
    Plotting stuff
    """

    def _set_up_figure(self):
        PLOTCOLS = 3
        PLOTROWS = 1
        FIGSIZE = (10, 5)
        DPI = 500
        # Create fig layout
        fig, axes = plt.subplots(PLOTROWS, PLOTCOLS, dpi=DPI, figsize=FIGSIZE, sharey=True,
                                gridspec_kw={'width_ratios': [1, 1, 1]})

        ax_s = axes[0]
        ax_s.set_ylabel('WTD (m)')
        ax_s.set_xlabel('S')

        ax_t = axes[1]
        ax_t.set_xlabel('$T (m^2 d^{-1})$')

        ax_D = axes[2]
        ax_D.set_xlabel('$D (m^2 d^{-1})$')

        return fig, axes

    def _plot_peat_physical_property(self, x, y, axis):
        axis.plot(x, y)

    def plot(self, zeta, dem, b):
        fig, (ax_S, ax_T, ax_D) = self._set_up_figure()

        S = self.storage_from_zeta(zeta, dem)
        T = self.transmissivity_from_zeta(zeta, dem, b)
        D = T/S

        ax_S.plot(S, zeta)
        ax_T.plot(T, zeta)
        ax_D.plot(D, zeta)


class ExponentialStorage(AbstractParameterization):
    """
    Old parameterization
    $$
    S = s_1 e^{s_2 \zeta}

    T = t_1 (e^{s_2 \zeta} - e^{-s_2 b})
    $$

    """

    def theta_from_zeta(self, zeta, dem):
        return self.s1/self.s2*(numerix.exp(self.s2*zeta) - numerix.exp(-self.s2*dem))

    def zeta_from_theta(self, theta, dem):
        return numerix.log(self.s2*theta/self.s1 + numerix.exp(-self.s2*dem))/self.s2

    def diffusion(self, theta, dem, b):
        s1 = self.s1
        s2 = self.s2
        t1 = self.t1
        t2 = self.t2

        return (
                t1/s1**(t2/s2)*(s1*numerix.exp(-dem*s2) + s2*theta)**(t2/s2-1) -
                t1*numerix.exp(-b*t2)/(s1*numerix.exp(-dem*s2) + s2*theta)
                )

    # Some optional functions:
    def h_from_theta(self, theta, dem):
        return numerix.log(self.s2/self.s1*numerix.exp(self.s2*dem)*theta + 1)/self.s2

    def theta_from_h(self, h, dem):
        return self.s1/self.s2*(numerix.exp(self.s2*(h-dem)) - numerix.exp(-self.s2*dem))

    def zeta_from_h(self, h, dem):
        return h - dem

    def h_from_zeta(self, zeta, dem):
        return zeta + dem

    """
    Plotting stuff
    """

    def storage_from_zeta(self, zeta, dem):
        import numpy as np
        s1 = self.s1
        s2 = self.s2

        return s1*np.exp(s2*zeta)

    def transmissivity_from_zeta(self, zeta, dem, b):
        import numpy as np

        t1 = self.t1
        t2 = self.t2

        return t1*(np.exp(t2*zeta) - np.exp(-t2*b))


class LinearBelowOneAboveStorage(AbstractParameterization):
    """
    Storage and transmissivity are piecewise functions with 2 pieces:
    below (zeta negative) and above (zeta positive) the surface.
    S and T were chosen so that D is continuous and differentiable at the surface.
    (This implies that T must be discontinuous, but that's fine!)

    Main characteristics of this parameterization:
    Above the surface:
        - S = 1
        - T = \alpha * \zeta + \beta
    Below the surface:
        - S = s_1 - s_2 * \zeta
        - T = t_1 * (e^{t_2*\zeta} - e^{-t_2*b})
    """

    def get_theta_knot(self, dem):
        return self.s1*dem + self.s2*dem**2/2

    def _get_piecewise_masks_in_theta(self, theta, dem):
        theta_knot = self.get_theta_knot(dem)
        mask_1 = 1*(theta >= theta_knot)
        mask_2 = 1 - mask_1

        return mask_1, mask_2

    def _get_piecewise_masks_in_zeta(self, zeta):
        positives_mask = 1*(zeta >= 0)
        negatives_mask = 1 - positives_mask

        return positives_mask, negatives_mask

    def _mask_array_piecewise(self, arr, mask_1, mask_2):
        return arr*mask_1, arr*mask_2

    def theta_from_zeta(self, zeta, dem):
        s1 = self.s1
        s2 = self.s2

        theta_knot = self.get_theta_knot(dem)
        positives_mask, negatives_mask = self._get_piecewise_masks_in_zeta(
            zeta)

        theta_pos = zeta + theta_knot
        theta_neg = s1*zeta - s2*zeta**2/2 + theta_knot

        return theta_pos*positives_mask + theta_neg*negatives_mask

    def zeta_from_theta(self, theta, dem):
        s1 = self.s1
        s2 = self.s2

        theta_knot = self.get_theta_knot(dem)
        mask_1, mask_2 = self._get_piecewise_masks_in_theta(theta, dem)

        theta_knot_1, theta_knot_2 = self._mask_array_piecewise(
            theta_knot, mask_1, mask_2)
        theta_1, theta_2 = self._mask_array_piecewise(theta, mask_1, mask_2)

        zeta_1 = theta_1 - theta_knot_1
        zeta_2 = s1/s2 - numerix.sqrt(s1**2 + 2*s2*(theta_knot_2 - theta_2))/s2

        return zeta_1*mask_1 + zeta_2*mask_2

    def diffusion(self, theta, dem, b):
        s1 = self.s1
        s2 = self.s2
        t1 = self.t1
        t2 = self.t2

        theta_knot = self.get_theta_knot(dem)

        mask_1, mask_2 = self._get_piecewise_masks_in_theta(theta, dem)

        theta_1, theta_2 = self._mask_array_piecewise(theta, mask_1, mask_2)
        dem_1, dem_2 = self._mask_array_piecewise(dem, mask_1, mask_2)
        b_1, b_2 = self._mask_array_piecewise(b, mask_1, mask_2)
        theta_knot_1, theta_knot_2 = self._mask_array_piecewise(
            theta_knot, mask_1, mask_2)

        sto_2 = numerix.sqrt(s1**2 + 2*s2*(theta_knot_2 - theta_2))

        alpha = t1*t2/s1**2 + s2*t1/s1**3 * (1 - numerix.exp(-t2*b_1))
        beta = t0/s1 + t1/s1*(1 - numerix.exp(-t2*b_1))

        diff_coeff_1 = alpha*(theta_1 - theta_knot_1) + \
                              beta  # Linear T above the surface

        diff_coeff_2 = t0/sto_2 + t1 * \
            (numerix.exp(t2/s2*(s1 - sto_2)) - numerix.exp(-t2*b))/sto_2

        diff_coeff = diff_coeff_1*mask_1 + diff_coeff_2*mask_2

        return diff_coeff

    """
    Plotting stuff
    """

    def storage_from_zeta(self, zeta, dem):
        s1 = self.s1
        s2 = self.s2

        positives_mask, negatives_mask = self._get_piecewise_masks_in_zeta(
            zeta)
        zeta_pos, zeta_neg = self._mask_array_piecewise(
            zeta, positives_mask, negatives_mask)

        sto_pos = 1
        sto_neg = (s1 - s2*zeta_neg)

        return sto_pos * positives_mask + sto_neg * negatives_mask

    def transmissivity_from_zeta(self, zeta, dem, b):
        import numpy as np

        s1 = self.s1
        s2 = self.s2
        t0 = self.t0
        t1 = self.t1
        t2 = self.t2

        positives_mask, negatives_mask = self._get_piecewise_masks_in_zeta(
            zeta)
        zeta_pos, zeta_neg = self._mask_array_piecewise(
            zeta, positives_mask, negatives_mask)

        alpha = t1*t2/s1**2 + s2*t1/s1**3 * (1 - numerix.exp(-t2*b))
        beta = t0/s1 + t1/s1*(1 - np.exp(-t2*b))

        tra_pos = alpha*zeta_pos + beta
        tra_neg = t0 + t1*(np.exp(t2*zeta) - np.exp(-t2*b))

        return tra_pos*positives_mask + tra_neg*negatives_mask


class ExponentialBelowOneAboveStorage(AbstractParameterization):
    """
    Storage and transmissivity are piecewise functions with 2 pieces:
    below (zeta negative) and above (zeta positive) the surface.
    S and T were chosen so that D is continuous and differentiable at the surface.
    (This implies that T must be discontinuous, but that's fine!)

    Main characteristics of this parameterization:
    Above the surface:
        - S = 1
        - T = \alpha * \zeta + \beta
    Below the surface:
        - S = s_1 e^{s2 * \zeta}
        - T = t_1 * e^{t_2*\zeta}
    """

    def get_theta_knot(self, dem):
        return self.s1/self.s2 * (1 - numerix.exp(-self.s2*dem))

    def _get_piecewise_masks_in_theta(self, theta, dem):
        theta_knot = self.get_theta_knot(dem)
        mask_1 = 1*(theta >= theta_knot)
        mask_2 = 1 - mask_1

        return mask_1, mask_2

    def _get_piecewise_masks_in_zeta(self, zeta):
        positives_mask = 1*(zeta >= 0)
        negatives_mask = 1 - positives_mask

        return positives_mask, negatives_mask

    def _mask_array_piecewise(self, arr, mask_1, mask_2):
        return arr*mask_1, arr*mask_2

    def theta_from_zeta(self, zeta, dem):
        s1 = self.s1
        s2 = self.s2

        theta_knot = self.get_theta_knot(dem)
        positives_mask, negatives_mask = self._get_piecewise_masks_in_zeta(
            zeta)

        theta_pos = zeta + theta_knot
        theta_neg = s1/s2*(numerix.exp(s2*zeta) - numerix.exp(-s2*dem))

        return theta_pos*positives_mask + theta_neg*negatives_mask

    def zeta_from_theta(self, theta, dem):
        s1 = self.s1
        s2 = self.s2

        theta_knot = self.get_theta_knot(dem)
        mask_1, mask_2 = self._get_piecewise_masks_in_theta(theta, dem)

        theta_knot_1, theta_knot_2 = self._mask_array_piecewise(
            theta_knot, mask_1, mask_2)
        theta_1, theta_2 = self._mask_array_piecewise(theta, mask_1, mask_2)

        zeta_1 = theta_1 - theta_knot_1
        zeta_2 = numerix.log(numerix.exp(-s2*dem) + s2/s1*theta)/s2

        return zeta_1*mask_1 + zeta_2*mask_2

    def diffusion(self, theta, dem, b):
        s1 = self.s1
        s2 = self.s2
        t1 = self.t1
        t2 = self.t2
        
        theta_knot = self.get_theta_knot(dem)
        
        mask_1, mask_2 = self._get_piecewise_masks_in_theta(theta, dem)

        theta_1, theta_2 = self._mask_array_piecewise(theta, mask_1, mask_2)
        dem_1, dem_2 = self._mask_array_piecewise(dem, mask_1, mask_2)
        b_1, b_2 = self._mask_array_piecewise(b, mask_1, mask_2)
        theta_knot_1, theta_knot_2 = self._mask_array_piecewise(theta_knot, mask_1, mask_2)

        sto_2 = s1*numerix.exp(-s2*dem) + s2*theta_2
        
        alpha = t1/s1**2*(t2 - s2)
        beta = t1/s1

        diff_coeff_1 = alpha*(theta_1 - theta_knot_1) + beta # Linear T above the surface
        # diff_coeff_1 = t1/s1 # constant diffusivity aboveground
        
        diff_coeff_2 = t1*sto_2**(t2/s2-1)/s1**(t2/s2)
        
        diff_coeff = diff_coeff_1*mask_1 + diff_coeff_2*mask_2

        return diff_coeff

    
    """
    Plotting stuff
    """
    
    def storage_from_zeta(self, zeta, dem):
        import numpy as np
        
        positives_mask = 1*(zeta>0)
        negatives_mask = 1 - positives_mask
        zeta_pos, zeta_neg = self._mask_array_piecewise(zeta, positives_mask, negatives_mask)

        sto_pos = 1
        sto_neg = self.s1*np.exp(self.s2*zeta)

        return sto_pos * positives_mask + sto_neg * negatives_mask


    def transmissivity_from_zeta(self, zeta, dem, b):
        import numpy as np
        
        s1 = self.s1
        s2 = self.s2
        t1 = self.t1
        t2 = self.t2

        positives_mask = 1*(zeta>0)
        negatives_mask = 1 - positives_mask
        zeta_pos, zeta_neg = self._mask_array_piecewise(zeta, positives_mask, negatives_mask)

        alpha = t1*t2
        beta = t1/s1
        
        tra_pos = alpha*zeta_pos + beta
        tra_neg = t1*np.exp(t2*zeta)

        return tra_pos*positives_mask + tra_neg*negatives_mask

class ExponentialBelowOneAboveStorageWithDepth(AbstractParameterization):
    """
    Storage and transmissivity are piecewise functions with 2 pieces:
    below (zeta negative) and above (zeta positive) the surface.
    S and T were chosen so that D is continuous and differentiable at the surface.
    (This implies that T must be discontinuous, but that's fine!)
    'WithDepth' means that the peat column depth is not neglected.

    Main characteristics of this parameterization:
    Above the surface:
        - S = 1
        - T = \alpha * \zeta + \beta
    Below the surface:
        - S = s_1 e^{s2 * \zeta}
        - T = t_1 * e^{t_2*\zeta}
    """

    def get_theta_knot(self, dem):
        return self.s1/self.s2 * (1 - numerix.exp(-self.s2*dem))

    def _get_piecewise_masks_in_theta(self, theta, dem):
        theta_knot = self.get_theta_knot(dem)
        mask_1 = 1*(theta >= theta_knot)
        mask_2 = 1 - mask_1

        return mask_1, mask_2

    def _get_piecewise_masks_in_zeta(self, zeta):
        positives_mask = 1*(zeta >= 0)
        negatives_mask = 1 - positives_mask

        return positives_mask, negatives_mask

    def _mask_array_piecewise(self, arr, mask_1, mask_2):
        return arr*mask_1, arr*mask_2

    def theta_from_zeta(self, zeta, dem):
        s1 = self.s1
        s2 = self.s2

        theta_knot = self.get_theta_knot(dem)
        positives_mask, negatives_mask = self._get_piecewise_masks_in_zeta(
            zeta)

        theta_pos = zeta + theta_knot
        theta_neg = s1/s2*(numerix.exp(s2*zeta) - numerix.exp(-s2*dem))

        return theta_pos*positives_mask + theta_neg*negatives_mask

    def zeta_from_theta(self, theta, dem):
        s1 = self.s1
        s2 = self.s2

        theta_knot = self.get_theta_knot(dem)
        mask_1, mask_2 = self._get_piecewise_masks_in_theta(theta, dem)

        theta_knot_1, theta_knot_2 = self._mask_array_piecewise(
            theta_knot, mask_1, mask_2)
        theta_1, theta_2 = self._mask_array_piecewise(theta, mask_1, mask_2)

        zeta_1 = theta_1 - theta_knot_1
        zeta_2 = numerix.log(numerix.exp(-s2*dem) + s2/s1*theta)/s2

        return zeta_1*mask_1 + zeta_2*mask_2

    def diffusion(self, theta, dem, b):
        s1 = self.s1
        s2 = self.s2
        t1 = self.t1
        t2 = self.t2
        
        theta_knot = self.get_theta_knot(dem)
        
        mask_1, mask_2 = self._get_piecewise_masks_in_theta(theta, dem)

        theta_1, theta_2 = self._mask_array_piecewise(theta, mask_1, mask_2)
        dem_1, dem_2 = self._mask_array_piecewise(dem, mask_1, mask_2)
        b_1, b_2 = self._mask_array_piecewise(b, mask_1, mask_2)
        theta_knot_1, theta_knot_2 = self._mask_array_piecewise(theta_knot, mask_1, mask_2)

        sto_2 = s1*numerix.exp(-s2*dem) + s2*theta_2
        
        alpha = t1/s1**2*(t2 - s2 + s2/numerix.exp(t2*b))
        beta = t1/s1 * (1 -numerix.exp(-t2*b) )
        
        depth_term = t1 * numerix.exp(-t2*b)/ sto_2

        diff_coeff_1 = alpha*(theta_1 - theta_knot_1) + beta # Linear T above the surface
        # diff_coeff_1 = t1/s1 # constant diffusivity aboveground
        
        diff_coeff_2 = t1*sto_2**(t2/s2-1)/s1**(t2/s2) - depth_term
        
        diff_coeff = diff_coeff_1*mask_1 + diff_coeff_2*mask_2

        return diff_coeff

    
    """
    Plotting stuff
    """
    
    def storage_from_zeta(self, zeta, dem):
        import numpy as np
        
        positives_mask = 1*(zeta>0)
        negatives_mask = 1 - positives_mask
        zeta_pos, zeta_neg = self._mask_array_piecewise(zeta, positives_mask, negatives_mask)

        sto_pos = 1
        sto_neg = self.s1*np.exp(self.s2*zeta)

        return sto_pos * positives_mask + sto_neg * negatives_mask


    def transmissivity_from_zeta(self, zeta, dem, b):
        import numpy as np
        
        s1 = self.s1
        s2 = self.s2
        t1 = self.t1
        t2 = self.t2

        positives_mask = 1*(zeta>0)
        negatives_mask = 1 - positives_mask
        zeta_pos, zeta_neg = self._mask_array_piecewise(zeta, positives_mask, negatives_mask)

        alpha = t1*t2
        beta = t1/s1
        
        tra_pos = alpha*zeta_pos + beta
        tra_neg = t1*(np.exp(t2*zeta) - np.exp(t2*b))

        return tra_pos*positives_mask + tra_neg*negatives_mask
