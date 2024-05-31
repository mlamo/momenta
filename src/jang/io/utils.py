import numpy as np

from jang.io import GW, NuDetector, Parameters


def get_search_region(detector: NuDetector, gw: GW, pars: Parameters):
    """Find the search region for the joint GW+nu signal based on a given GW event, list of neutrino samples and configuration>
    In the configuration, the entry `search_region` is used to define the search region with respect """

    region = gw.fits.get_signal_region(pars.nside, pars.get_searchregion_gwfraction())
    if not pars.get_searchregion_iszeroincluded():
        region_nonzero = detector.get_nonempty_acceptance_pixels(pars.flux, pars.nside)
        region = np.intersect1d(region, region_nonzero)
    if len(region) == 0:
        raise RuntimeError("The search region has been reduced to empty. Please check 'search_region' parameter.")
    return region
