import abc
import numpy as np
import pymc as pm
from scipy.integrate import quad

import jang.utils.conversions

Mpc_to_cm = 3.0856776e24
erg_to_GeV = 624.15


class Component(abc.ABC):
    
    def __init__(self, emin, emax):
        self.emin = emin
        self.emax = emax
        pass
    
    @abc.abstractmethod  # pragma: no cover    
    def evaluate(self, energy):
        return None
    
    def __eq__(self, other):
        return self.__dict__ == other.__dict__
    def __str__(self):
        return type(self).__name__
    def __hash__(self):
        return hash(self.__str__())

    def flux_to_eiso(self, distance_Mpc):
        distance_cm = distance_Mpc * Mpc_to_cm
        f = lambda x: self.evaluate(np.exp(x)) * (np.exp(x))**2
        integration = quad(f, np.log(self.emin), np.log(self.emax), limit=100)[0]
        return integration * (4 * np.pi * distance_cm**2) / erg_to_GeV
    
    
class PowerLaw(Component):
    
    def __init__(self, emin, emax, spectral_index, eref=1):
        super().__init__(emin=emin, emax=emax)
        self.spectral_index = spectral_index
        self.ncomponents = 1
        self.eref = eref
        
    def __str__(self):
        return f"{type(self).__name__}/gamma={self.spectral_index}/{self.emin}--{self.emax}"
    
    def evaluate(self, energy):
        return np.where((self.emin <= energy) & (energy <= self.emax), np.power(energy/self.eref, -self.spectral_index), 0)
    
    
class FluxBase(abc.ABC):
    
    def __init__(self):
        self.components = []
       
    @property 
    def ncomponents(self):
        return len(self.components)
      
    def evaluate(self, energy):
        return [c.evaluate(energy) for c in self.components]
 
    def define_signal_parameters(self):
        self.norm = pm.Uniform("phi", lower=0, upper=1e9, dims=("fluxcomponents",))
        return self.norm

    def define_expected_signal(self, gw, itoygw, xacc, detector, nside):
        ipix = pm.Deterministic("ipix", gw[itoygw][0]).astype("int64")
        acc = pm.Data("acc", [detector.get_acceptances(c, nside) for c in self.components], dims=("fluxcomponents", "nusample", "ipix"))
        sig = pm.Deterministic("sig", 1/6 * self.norm.dot(xacc * acc[:,:,ipix]), dims="nusample")
        return sig
    
    def define_auxiliary(self, gw, itoygw, parameters):
        eiso = pm.Deterministic("eiso", self.norm.dot([c.flux_to_eiso(gw[itoygw][1]) for c in self.components]))
        etot = pm.Deterministic("etot", eiso / jang.utils.conversions.etot_to_eiso(gw[itoygw][3], parameters.jet))
        fnu = pm.Deterministic("fnu", etot / jang.utils.conversions.fnu_to_etot(gw[itoygw][2]))
        return eiso, etot, fnu


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