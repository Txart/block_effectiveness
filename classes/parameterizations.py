import matplotlib.pyplot as plt
from fipy import numerix


from classes.peat_hydro_params import PeatlandHydroParameters


class AbstractParameterization:
    def __init__(self,
                 ph_params:PeatlandHydroParameters) -> None:

        self.s1 = ph_params.s1
        self.s2 = ph_params.s2
        self.t1 = ph_params.t1
        self.t2 = ph_params.t2


        pass

    """
    Allows 2 choices for S:
    - S = constant below, 1 above; by setting s2 = 0
    - S = exponential below, 1 above; by setting s2 != 0

    Those choices will be fixed in the subclasses

    T linear aboveground, exponential belowground

    This class is abstract. Always create one of its subclasses.
    Here I define the functions that every subclass of this class
    should have.
    """

    def zeta_from_h(self, h, dem):
        return h - dem

    def h_from_zeta(self, zeta, dem):
        return zeta + dem



    def _get_piecewise_masks_in_h(self, h, dem):
        mask_1 = 1 * (h >= dem)
        mask_2 = 1 - mask_1

        return mask_1, mask_2

    def _get_piecewise_masks_in_zeta(self, zeta):
        positives_mask = 1*(zeta >= 0)
        negatives_mask = 1 - positives_mask

        return positives_mask, negatives_mask

    def _mask_array_piecewise(self, arr, mask_1, mask_2):
        return arr*mask_1, arr*mask_2


    def storage(self, h, dem):
        s1 = self.s1
        s2 = self.s2

        return self.s1 * numerix.exp(self.s2*(h - dem))


    def transmissivity(self, h, dem, depth):
        t1 = self.t1
        t2 = self.t2
        
        mask_1, mask_2 = self._get_piecewise_masks_in_h(h, dem)

        # h_1, h_2 = self._mask_array_piecewise(h, mask_1, mask_2)
        # dem_1, dem_2 = self._mask_array_piecewise(dem, mask_1, mask_2)
        # depth_1, depth_2 = self._mask_array_piecewise(depth, mask_1, mask_2)
        
        trans_1 = t1*( 1 + t2*(h - dem) - numerix.exp(-t2*depth))
        trans_2 = t1*( numerix.exp(t2*(h - dem)) - numerix.exp(-t2 * depth))
        
        # return trans_1*mask_1 + trans_2*mask_2
        return trans_2

    
    """
    Not necessary for hydro, but useful for
    Plotting stuff
    """

    def storage_from_zeta(self, zeta):
        import numpy as np # importing inside function not to contaminate fipy's numerix.
        
        return self.s1*np.exp(self.s2*zeta)



    def transmissivity_from_zeta(self, zeta, depth):
        import numpy as np # importing inside function not to contaminate fipy's numerix.
        
        t1 = self.t1
        t2 = self.t2
        
        positives_mask = 1*(zeta>0)
        negatives_mask = 1 - positives_mask
        zeta_pos, zeta_neg = self._mask_array_piecewise(zeta, positives_mask, negatives_mask)

        trans_pos = t1*(1 + t2*zeta_pos - np.exp(-t2*depth))
        trans_neg = t1*(np.exp(t2*zeta_neg) - np.exp(-t2*depth))

        return trans_pos * positives_mask + trans_neg * negatives_mask


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

    def plot(self, zeta, dem, depth):
        fig, (ax_S, ax_T, ax_D) = self._set_up_figure()

        S = self.storage_from_zeta(zeta)
        T = self.transmissivity_from_zeta(zeta, depth)
        D = T/S

        ax_S.plot(S, zeta)
        ax_T.plot(T, zeta)
        ax_D.plot(D, zeta)


class ConstantStorage(AbstractParameterization):
    """
    S = 1 aboveground,
    S = s1 belowground. This is obtained by hard-setting s2=0.

    """

    def __init__(self, ph_params:PeatlandHydroParameters) -> None:
        super().__init__(ph_params)

        self.s2 = 0

        pass


class ExponentialStorage(AbstractParameterization):
    """
    This doesn't need any change from the main class!
    """
    def __init__(self, ph_params: PeatlandHydroParameters) -> None:
        super().__init__(ph_params)

        pass

