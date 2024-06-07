import abc
import numpy as np
import pymc as pm
from scipy.integrate import quad

import jang.utils.conversions

Mpc_to_cm = 3.0856776e24
erg_to_GeV = 624.15


class Component(abc.ABC):
    
    def __init__(self, emin, emax, store_acceptance=True):
        self.emin = emin
        self.emax = emax
        self.store_acceptance = store_acceptance
        self.shape_names = []
        self.shape_pars = []
        pass
    
    def set_shape(self, shape):
        self.shape_pars = shape
    
    @abc.abstractmethod  # pragma: no cover    
    def evaluate(self, energy):
        return None
    
    def __eq__(self, other):
        return self.__dict__ == other.__dict__
    def __str__(self):
        return type(self).__name__
    def __hash__(self):
        return hash(self.__str__())

    def flux_to_eiso(self, distance_scaling):
        f = lambda x: self.evaluate(np.exp(x)) * (np.exp(x))**2
        integration = quad(f, np.log(self.emin), np.log(self.emax), limit=100)[0]
        return distance_scaling * integration
    
    
class PowerLaw(Component):
    
    def __init__(self, emin, emax, spectral_index, eref=1, store_acceptance=True):
        super().__init__(emin=emin, emax=emax, store_acceptance=store_acceptance)
        self.shape_names = ["gamma", "eref"]
        self.shape_pars = [spectral_index, eref]
        
    def __str__(self):
        strshape = "/".join([f"{n}={p}" for n, p in zip(self.shape_names, self.shape_pars)])
        return f"{type(self).__name__}/{self.emin}--{self.emax}/{strshape}"
    
    def evaluate(self, energy):
        return np.where((self.emin <= energy) & (energy <= self.emax), np.power(energy/self.shape_pars[1], -self.shape_pars[0]), 0)


class FluxBase(abc.ABC):
    
    def __init__(self):
        self.components = []
       
    @property 
    def ncomponents(self):
        return len(self.components)
      
    def evaluate(self, energy):
        return [c.evaluate(energy) for c in self.components]

    def flux_to_eiso(self, distance_scaling):
        return np.array([c.flux_to_eiso(distance_scaling) for c in self.components])


class FluxFixedPowerLaw(FluxBase):

    def __init__(self, emin, emax, spectral_index, eref=1):
        super().__init__()
        self.components = [PowerLaw(emin, emax, spectral_index, eref)]


class FluxFixedDoublePowerLaw(FluxBase):

    def __init__(self, emin, emax, spectral_indices, eref=1):
        super().__init__()
        self.components = [
            PowerLaw(emin, emax, spectral_indices[0], eref),
            PowerLaw(emin, emax, spectral_indices[1], eref),
        ]