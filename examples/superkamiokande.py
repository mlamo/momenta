"""Example to handle Super-Kamiokande specific format.

In this case, the input from Super-K are simply the number of events (observed and expected),
as well as the detector effective area in the format of 2D histograms with x=log10(energy [in GeV]) and y=zenith angle [in rad].
The provided values in this example are all dummy, except for the effective area that is the one published in https://doi.org/10.5281/zenodo.4724822.
"""

import astropy.time
import h5py
from scipy.interpolate import RegularGridInterpolator

import jang.utils.flux as flux
from jang.io import GWDatabase, NuDetector, Parameters
from jang.io.neutrinos import EffectiveAreaAltitudeDep, BackgroundFixed
from jang.stats.run import run
from jang.stats.limits import get_limits


class EffectiveAreaSK(EffectiveAreaAltitudeDep):
    def __init__(self, filename: str, samplename: str, time: astropy.time.Time):
        super().__init__()
        self.read(filename, samplename)
        self.set_location(time, 36.425634, 137.310340)

    def read(self, filename: str, samplename: str):
        with h5py.File(filename, "r") as f:
            bins_logenergy = f["bins_logenergy"][:]
            bins_altitude = f["bins_altitude"][:]
            aeff = f[f"aeff_{samplename}"][:]
        self.func = RegularGridInterpolator((bins_logenergy, bins_altitude), aeff, bounds_error=False, fill_value=0)


def single_event(gwname: str, gwdbfile: str, det_results: dict, pars: Parameters):
    """Compute the limits for a given GW event and using the detector results stored in dictionary.

    The `det_results` dictionary should contain the following keys:
        - nobs: list of observed number of events (length = 4 [number of samples])
        - nbkg: list of expected number of events (length = 4 [number of samples])
        - effarea: path to the effective area file
    """

    database_gw = GWDatabase(gwdbfile)
    database_gw.set_parameters(pars)

    sk = NuDetector("examples/input_files/detector_superk.yaml")
    gw = database_gw.find_gw(gwname)

    sk.set_effective_areas([EffectiveAreaSK(det_results["effarea"], s.name, gw.utc) for s in sk.samples])
    bkg = [BackgroundFixed(b) for b in det_results["nbkg"]]
    sk.set_observations(det_results["nobs"], bkg)

    model, result = run(sk, gw, pars, method="ultranest")
    limits = get_limits(result["samples"], model, CL=0.90)
    print("90% upper limit on flux normalisation:", limits["flux0_norm"])


if __name__ == "__main__":

    parameters = Parameters("examples/input_files/config.yaml")
    parameters.set_models(flux=flux.FluxFixedPowerLaw(0.1, 1e6, 2))
    parameters.nside = 8

    gwdb = "examples/input_files/gw_catalogs/database_example.csv"
    detresults = {
        "nobs": [0, 0, 0],
        "nbkg": [0.112, 0.007, 0.016],
        "effarea": "examples/input_files/effarea_superk.h5",
    }
    single_event("GW190412", gwdb, detresults, parameters)
