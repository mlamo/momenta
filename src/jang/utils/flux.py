import abc
import numpy as np
import pymc as pm
from scipy.integrate import quad

from jang.io.neutrinos import EffectiveAreaBase
import jang.utils.conversions

Mpc_to_cm = 3.0856776e24
erg_to_GeV = 624.15


class FluxBase(abc.ABC):
    
    def __init__(self, emin, emax):
        self.emin = emin
        self.emax = emax
        pass
    
    @abc.abstractmethod  # pragma: no cover    
    def evaluate(self, energy):
        return None
    
    def __eq__(self, other):
        return self.__dict__ == other.__dict__
    def __hash__(self):
        return hash((type(self).__name__,) + tuple(self.__dict__))

    @abc.abstractmethod  # pragma: no cover    
    def define_signal_parameters(self, gw, itoygw, parameters):
        pass

    @abc.abstractmethod  # pragma: no cover    
    def define_expected_signal(self, ipix, xacc, aeffs):
        pass

    def get_acceptance(self, aeff: EffectiveAreaBase, *args):
        def func(x: float, *args):
            return self.evaluate(np.exp(x)) * aeff.evaluate(np.exp(x), *args) * np.exp(x)
        return quad(func, np.log(self.emin), np.log(self.emax), limit=500, args=args)[0]
        
        
class FixedPowerLaw(FluxBase):
    
    def __init__(self, emin, emax, spectral_index, eref=1):
        super().__init__(emin=emin, emax=emax)
        self.spectral_index = spectral_index
        self.eref = eref
        
    def evaluate(self, energy):
        return np.where((self.emin <= energy) & (energy <= self.emax), np.power(energy/self.eref, -self.spectral_index), 0)

    def flux_to_eiso(self, distance_Mpc):
        distance_cm = distance_Mpc * Mpc_to_cm
        f = lambda x: self.evaluate(np.exp(x)) * (np.exp(x))**2
        integration = quad(f, np.log(self.emin), np.log(self.emax), limit=100)[0]
        return integration * (4 * np.pi * distance_cm**2) / erg_to_GeV
    
    def define_signal_parameters(self, gw, itoygw, parameters):
        phi = pm.Uniform("phi", lower=0, upper=1e9)
        eiso = pm.Deterministic("eiso", phi * self.flux_to_eiso(gw[itoygw][1]))
        etot = pm.Deterministic("etot", eiso / jang.utils.conversions.etot_to_eiso(gw[itoygw][3], parameters.jet))
        fnu = pm.Deterministic("fnu", etot / jang.utils.conversions.fnu_to_etot(gw[itoygw][2]))
        self.signal_main = (phi,)
        self.signal_aux = (eiso, etot, fnu)
        return self.signal_main, self.signal_aux

    def define_expected_signal(self, ipix, xacc, aeffs):
        acc = pm.Deterministic("acc", xacc * [self.get_acceptance(aeff) for aeff in aeffs], dims="nusample")
        sig = pm.Deterministic("sig", self.signal_main[0] / 6 * acc, dims="nusample")
        return sig
    