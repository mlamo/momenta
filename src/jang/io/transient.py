import abc
import astropy.coordinates
import astropy.time
import astropy.units as u
import healpy as hp
import logging
import pandas as pd

import jang.utils.conversions


class Transient:
    
    def __init__(self, name: str = None, utc: astropy.time.Time|None = None, logger: str = "jang"):
        self.name = name
        self.utc = utc
        self.logger = logger
       
    @property 
    def log(self):
        return logging.getLogger(self.logger)
        
    @abc.abstractmethod
    def prepare_prior_samples(self) -> pd.DataFrame:
        return
    
    
class PointSource(Transient):
    
    def __init__(self, ra_deg: float, dec_deg: float, err_deg: float, name: str = None, utc: astropy.time.Time|None = None, logger: str = "jang"):
        super().__init__(name, utc, logger)
        self.coords = astropy.coordinates.SkyCoord(ra=ra_deg*u.deg, dec=dec_deg*u.deg, frame="icrs")
        self.err = err_deg * u.deg
        self.distance = None
        self.redshift = None
        
    def set_distance(self, distance):
        self.distance = distance
        self.redshift = jang.utils.conversions.lumidistance_to_redshift(distance)
        
    def set_redshift(self, redshift):
        self.distance = jang.utils.conversions.redshift_to_lumidistance(redshift)
        self.redshift = redshift
        
    def prepare_prior_samples(self, nside: int) -> pd.DataFrame:
        toys = {}
        if self.err == 0 * u.deg:
            toys["ra"] = [self.coords.ra.deg]
            toys["dec"] = [self.coords.dec.deg]
            if self.distance:
                toys["distance_scaling"] = [jang.utils.conversions.distance_scaling(self.distance, self.redshift)]
        else:
            raise RuntimeError("Not implemented yet")
        toys["ipix"] = hp.ang2pix(nside, toys["ra"], toys["dec"], lonlat=True)
        return pd.DataFrame(data=toys)