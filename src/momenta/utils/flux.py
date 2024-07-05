"""
    Copyright (C) 2024  Mathieu Lamoureux

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import abc
import numpy as np
from scipy.integrate import quad


class Component(abc.ABC):
    
    def __init__(self, emin, emax, store_acceptance=True):
        self.emin = emin
        self.emax = emax
        self.store_acceptance = store_acceptance
        self.shape_names = []
        self.shape_values = []
        self.shape_defaults = []
        self.shape_boundaries = []
        
    def __str__(self):
        s = f"{type(self).__name__}/{self.emin:.1e}--{self.emax:.1e}"
        if len(self.shape_names) == 0:
            return s
        strshape = "/".join([f"{n}={v}" for n, v in zip(self.shape_names, self.shape_values)])
        return s + "/" + strshape
    
    def __eq__(self, other):
        return self.__dict__ == other.__dict__
    
    def __hash__(self):
        return hash(self.__str__())

    @property
    def nshapes(self):
        return len(self.shape_names)
    
    def set_shapes(self, shapes):
        self.shape_values = shapes
    
    @abc.abstractmethod
    def evaluate(self, energy):
        return None

    def flux_to_eiso(self, distance_scaling):
        f = lambda x: self.evaluate(np.exp(x)) * (np.exp(x))**2
        integration = quad(f, np.log(self.emin), np.log(self.emax), limit=100)[0]
        return distance_scaling * integration
    
    def prior_transform(self, x):
        """Transform uniform parameters in [0, 1] to shape parameter space."""
        return x


class PowerLaw(Component):

    def __init__(self, emin, emax, eref=1):
        super().__init__(emin=emin, emax=emax, store_acceptance=False)
        self.shape_names = ["gamma"]
        self.shape_defaults = [2]
        self.shape_values = self.shape_defaults
        self.shape_boundaries = [(2, 3)]
        self.eref = eref

    def evaluate(self, energy):
        return np.where((self.emin <= energy) & (energy <= self.emax), np.power(energy/self.eref, -self.shape_values[0]), 0)

    def prior_transform(self, x):
        return self.shape_boundaries[0][0] + (self.shape_boundaries[0][1] - self.shape_boundaries[0][0]) * x


class FixedPowerLaw(Component):
    
    def __init__(self, emin, emax, spectral_index, eref=1):
        super().__init__(emin=emin, emax=emax, store_acceptance=True)
        self.spectral_index = spectral_index
        self.eref = eref
        
    def __str__(self):
        return f"{super().__str__()}/gamma={self.spectral_index}"
    
    def evaluate(self, energy):
        return np.where((self.emin <= energy) & (energy <= self.emax), np.power(energy/self.eref, -self.spectral_index), 0)


class FluxBase(abc.ABC):
    
    def __init__(self):
        self.components = []
        
    def __str__(self):
        return '__'.join([str(c) for c in self.components])

    @property 
    def ncomponents(self):
        return len(self.components)
    
    @property 
    def nshapes(self):
        return np.sum([c.nshapes for c in self.components])
    
    @property
    def nparameters(self):
        return self.ncomponents + self.nshapes
    
    @property
    def shape_positions(self):
        return np.cumsum([c.nshapes for c in self.components]).astype(int)

    @property
    def shape_defaults(self):
        defaults = []
        for c in self.components:
            defaults += list(c.shape_defaults)
        return defaults
    
    @property
    def shape_boundaries(self):
        boundaries = []
        for c in self.components:
            boundaries += list(c.shape_boundaries)
        return boundaries

    def set_shapes(self, shapes):
        for c, i in zip(self.components, self.shape_positions):
            c.set_shapes(shapes[i-c.nshapes:i])
            
    def evaluate(self, energy):
        return [c.evaluate(energy) for c in self.components]

    def flux_to_eiso(self, distance_scaling):
        return np.array([c.flux_to_eiso(distance_scaling) for c in self.components])
        
    def prior_transform(self, x):
        return [y for c, i in zip(self.components, self.shape_positions) for y in c.prior_transform(x[i-c.nshapes:i])]


class FluxFixedPowerLaw(FluxBase):

    def __init__(self, emin, emax, spectral_index, eref=1):
        super().__init__()
        self.components = [FixedPowerLaw(emin, emax, spectral_index, eref)]


class FluxVariablePowerLaw(FluxBase):

    def __init__(self, emin, emax, eref=1):
        super().__init__()
        self.components = [PowerLaw(emin, emax, eref)]


class FluxFixedDoublePowerLaw(FluxBase):

    def __init__(self, emin, emax, spectral_indices, eref=1):
        super().__init__()
        self.components = [
            FixedPowerLaw(emin, emax, spectral_indices[0], eref),
            FixedPowerLaw(emin, emax, spectral_indices[1], eref),
        ]
