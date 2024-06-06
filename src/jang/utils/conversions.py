"""Utility functions to perform conversions."""

import abc
import astropy.cosmology.units as acu
import astropy.time
import astropy.units as u
import datetime
import numpy as np
from typing import List

Mpc_to_cm = 3.0856776e24
erg_to_GeV = 624.15
solarmass_to_erg = 1.787e54
second_to_day = 1 / 86400


class JetModelBase(metaclass=abc.ABCMeta):
    """Abstract model for neutrino emission jetting."""

    def __init__(self, jet_opening: float):
        self.jet_opening = jet_opening

    @abc.abstractmethod
    def etot_to_eiso(self, viewing_angle: float):
        """Conversion to total energy to the equivalent isotropic energy."""
        pass

    @property
    def str_filename(self):
        return self.__repr__().lower().replace(",", "_").replace(" ", "")


class JetIsotropic(JetModelBase):
    """Isotropic emission of neutrinos."""

    def __init__(self):
        super().__init__(np.inf)

    def etot_to_eiso(self, viewing_angle: float) -> float:
        return 1

    def __repr__(self):
        return "Isotropic"


class JetVonMises(JetModelBase):
    """Emission in a Gaussian jet."""

    def __init__(self, jet_opening: float, with_counter: bool = False):
        super().__init__(jet_opening)
        self.with_counter = with_counter
        self.kappa = np.longdouble(1 / (self.jet_opening**2))

    def etot_to_eiso(self, viewing_angle: float) -> float:
        if np.isinf(self.jet_opening):
            return 1
        if self.with_counter:
            return self.kappa * np.cosh(self.kappa * np.cos(viewing_angle)) / np.sinh(self.kappa)
        return self.kappa * np.exp(self.kappa * np.cos(viewing_angle)) / np.sinh(self.kappa)

    def __repr__(self):
        return "VonMises,%.1f deg%s" % (np.rad2deg(self.jet_opening), ",w/counter" if self.with_counter else "")


class JetRectangular(JetModelBase):
    """Emission in a rectangular jet (constant inside a cone, zero elsewhere)."""

    def __init__(self, jet_opening: float, with_counter: bool = False):
        super().__init__(jet_opening)
        self.with_counter = with_counter

    def etot_to_eiso(self, viewing_angle: float) -> float:
        if not 0 < self.jet_opening < np.pi:
            return 1
        if self.with_counter:
            if viewing_angle <= self.jet_opening or viewing_angle >= np.pi - self.jet_opening:
                return 1 / (1 - np.cos(self.jet_opening))
            return 0
        if viewing_angle <= self.jet_opening:
            return 2 / (1 - np.cos(self.jet_opening))
        return 0

    def __repr__(self):
        return "Constant,%.1f deg%s)" % (self.jet_opening, ",w/counter" if self.with_counter else "")


def list_jet_models() -> List[JetModelBase]:
    """List all available jet models, with a scanning in opening angles."""
    full_list = []
    full_list.append(JetIsotropic())
    for opening in range(5, 60 + 1, 5):
        for with_counter in (False, True):
            full_list.append(JetVonMises(opening, with_counter=with_counter))
            full_list.append(JetRectangular(opening, with_counter=with_counter))
    return full_list


def etot_to_eiso(viewing_angle: float, model: JetModelBase) -> float:
    """Convert from total energy to the equivalent isotropic energy for a given jet model and viewing angle."""
    return model.etot_to_eiso(viewing_angle)


def fnu_to_etot(radiated_energy_gw: float) -> float:
    """Convert from fraction of radiated energy to total energy."""
    return radiated_energy_gw * solarmass_to_erg


def utc_to_jd(dtime: datetime.datetime) -> float:
    """Convert from UTC time (datetime format) to julian date."""
    return astropy.time.Time(dtime, format="datetime").jd


def jd_to_mjd(jd: float) -> float:
    """Convert Julian Date to Modified Julian Date."""
    return jd - 2400000.5


def redshift_to_lumidistance(redshift: float):
    return (redshift * acu.redshift).to(u.Mpc, acu.redshift_distance(kind="luminosity"))
    

def lumidistance_to_redshift(distance: float):
    return (distance * u.Mpc).to(acu.redshift, acu.redshift_distance(kind="luminosity"))


def distance_scaling(distance: float, redshift: float|None = None):
    """Returns the factor to scale from flux [/GeV/cm^2] to isotropic energy [erg]"""
    f = 4 * np.pi
    f *= ((distance*u.Mpc).to(u.cm).value)**2  # distance in cm
    if redshift is not None:
        f *= 1 / (1+redshift)
    else:
        f *= 1 / (1+lumidistance_to_redshift(distance))
    f *= (1*u.GeV).to(u.erg).value  # energy in erg
    return f