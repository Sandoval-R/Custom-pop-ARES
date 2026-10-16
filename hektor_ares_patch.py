"""
Monkey-patch
It helps connecting Hektor's PBH accertion model to ARES.
ARES usually callas the heating and ionization function with z.
The current code uses also T_k to calculate v_eff, so we need to change
f(z) -> f(z,T_k).

If we have ARES handling all the physics, we can remove this.
"""

import numpy as np
import ares.populations.Parameterized as _param_mod
import ares.static.VolumeGlobal as _vg_mod


from ares.util.ParameterFile import get_pq_pars
from ares.phenom.ParameterizedQuantity import ParameterizedQuantity
from scipy.interpolate import interp1d

_installed = False

def install_stateful_callbacks():
    """
    Install the patch once
    """
    global _installed
    if _installed:
        return
    _installed = True

    ParametricPopulation = _param_mod.ParametricPopulation  # ARES population
    GlobalVolume = _vg_mod.GlobalVolume                     # THE Global Volume calculation 

    def fixed_getattr(self, name):
        if name.startswith('_'):
            raise AttributeError(name)

        full_name = 'pop_' + name
        if name in self.__dict__:
            return self.__dict__[name]

        if full_name not in self.pf:
            raise AttributeError(name)

        value = self.pf[full_name]

        if isinstance(value, (float, np.floating)):
            result = lambda z, _value=value: _value
        elif callable(value):
            result = value
        elif isinstance(value, str) and value.startswith('pq'):
            result = ParameterizedQuantity(**get_pq_pars(value, self.pf))
        elif isinstance(value, str):
            x, y = np.loadtxt(value, unpack=True)
            result = interp1d(x, y, kind=self.pf['interp_hist'])
        else:
            raise NotImplementedError(f'Problem with: {name}')

        setattr(self, name, result)
        return result

    ParametricPopulation.__getattr__ = fixed_getattr

    def stateful_ionization(self, z, **kwargs):
        callback = self.pf.get('pop_ion_rate_igm')
        if callback is None:
            raise AttributeError('pop_ion_rate_igm')
        if getattr(callback, '_ares_stateful', False): # This is how we make sure that only the hektor's population gets the fix
            return callback(z=z, **kwargs)
        return callback(z=z)

    def stateful_heating(self, z, **kwargs):
        callback = self.pf.get('pop_heat_rate')
        if callback is None:
            raise AttributeError('pop_heat_rate')
        if getattr(callback, '_ares_stateful', False): # This is how we make sure that only the hektor's population gets the fix
            return callback(z=z, **kwargs)
        return callback(z=z)

    ParametricPopulation.IonizationRateIGM = stateful_ionization
    ParametricPopulation.HeatingRate = stateful_heating

    original_ionization_rate_igm = GlobalVolume.IonizationRateIGM
    original_heating_rate = GlobalVolume.HeatingRate

    def stateful_ionization_rate_igm(self, z, species=0, popid=0, band=0, **kwargs):
        pop = self.pops[popid]
        callback = pop.pf.get('pop_ion_rate_igm')
        if getattr(callback, '_ares_stateful', False):
            if (not pop.is_src_ion_igm) or (z > pop.zform):
                return 0.0
            return pop.IonizationRateIGM(
                z, species=species, popid=popid, band=band, **kwargs
            )
        return original_ionization_rate_igm(
            self, z, species=species, popid=popid, band=band, **kwargs
        )

    def stateful_heating_rate(self, z, species=0, popid=0, band=0, **kwargs):
        pop = self.pops[popid]
        callback = pop.pf.get('pop_heat_rate')
        if getattr(callback, '_ares_stateful', False):
            if (not pop.is_src_heat_igm) or (z >= pop.zform):
                return 0.0
            return pop.HeatingRate(
                z, species=species, popid=popid, band=band, **kwargs
            )
        return original_heating_rate(
            self, z, species=species, popid=popid, band=band, **kwargs
        )

    GlobalVolume.IonizationRateIGM = stateful_ionization_rate_igm
    GlobalVolume.HeatingRate = stateful_heating_rate


__all__ = ["install_stateful_callbacks"]
